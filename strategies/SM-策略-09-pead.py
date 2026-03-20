"""
PEAD（盈余公告后漂移）· 无预期数据下的量价异常事件驱动
策略与引擎分离：本模块只负责生成信号与持仓状态，不涉及资金扣减。

核心逻辑：
- 事件日 T = ann_date 的下一个交易日（市场对财报做出反应的第一天）
- 仅在 T 日判断：大涨 + 巨量 + 前期未抢跑 → entry_signal = 1
- 持仓 40 日或跌破 T 日开盘价 → exit_signal = 1
- 全程 Polars 向量化，严禁 for 循环遍历日期/股票
"""
from __future__ import annotations

import os
from datetime import datetime

import polars as pl

# 可调参数（便于回测时覆盖）
VOL_MA_WINDOW = 20        # 量比分母：过去 20 日均量
PRE_RET_WINDOW = 20       # 财报前 20 日累计涨幅（防内幕泄漏）
DAILY_RET_TH = 0.05       # T 日涨幅阈值（跳空/大涨视为利好）
VOL_RATIO_TH = 2.0       # 量比阈值（巨量扫货）
PRE_RET_CAP = 0.15       # 前期涨幅上限（排除利好出尽）
HOLD_DAYS_EXIT = 40      # 持仓满 40 个交易日退出


def generate_pead_signals(
    df_price: pl.DataFrame,
    df_ann: pl.DataFrame,
    **params,
) -> pl.DataFrame:
    """
    基于财报披露日的量价异常生成 PEAD 事件驱动信号。

    输入：
    - df_price: ts_code, trade_date, open, close, vol
    - df_ann: ts_code, ann_date（财报实际披露日）

    输出：原表 + daily_ret, vol_ma20, vol_ratio, pre_20d_ret, is_event_day,
          entry_signal, exit_signal, position_status
    """
    vol_ma_w = int(params.get("VOL_MA_WINDOW", VOL_MA_WINDOW))
    pre_w = int(params.get("PRE_RET_WINDOW", PRE_RET_WINDOW))
    daily_ret_th = float(params.get("DAILY_RET_TH", DAILY_RET_TH))
    vol_ratio_th = float(params.get("VOL_RATIO_TH", VOL_RATIO_TH))
    pre_ret_cap = float(params.get("PRE_RET_CAP", PRE_RET_CAP))
    hold_days = int(params.get("HOLD_DAYS_EXIT", HOLD_DAYS_EXIT))

    need_price = ["ts_code", "trade_date", "open", "close", "vol"]
    need_ann = ["ts_code", "ann_date"]
    for name, cols in [("df_price", need_price), ("df_ann", need_ann)]:
        missing = [c for c in cols if c not in (df_price.columns if name == "df_price" else df_ann.columns)]
        if missing:
            raise KeyError(f"{name} 缺少列: {missing}")

    # -------------------------------------------------------------------------
    # 1) 事件日 T = ann_date 的下一个交易日（每只股票、每条公告一个 T）
    #    关键点：避免 (ann x dates) 的笛卡尔爆炸，改用 asof “向前匹配”找下一交易日
    # -------------------------------------------------------------------------
    price_dates = (
        df_price.select(["ts_code", "trade_date"])
        .unique()
        .sort(["ts_code", "trade_date"])
        .with_columns(pl.col("trade_date").alias("_td"))
    )
    ann = (
        df_ann.select(["ts_code", "ann_date"])
        .unique()
        .sort(["ts_code", "ann_date"])
        .with_columns(pl.col("ann_date").alias("_ad"))
    )

    # Polars 版本兼容：优先使用 allow_exact_matches=False，确保严格 “下一交易日”
    try:
        event_T = (
            ann.join_asof(
                price_dates,
                left_on="_ad",
                right_on="_td",
                by="ts_code",
                strategy="forward",
                allow_exact_matches=False,
            )
            .select(["ts_code", "ann_date", pl.col("trade_date").alias("T_date")])
            .drop_nulls(["T_date"])
        )
    except TypeError:
        # 若不支持 allow_exact_matches：先取 >= 的匹配，遇到相等则用下一交易日
        next_td = pl.col("trade_date").shift(-1).over("ts_code").alias("_next_td")
        pd2 = price_dates.with_columns([next_td])
        tmp = (
            ann.join_asof(
                pd2,
                left_on="_ad",
                right_on="_td",
                by="ts_code",
                strategy="forward",
            )
            .with_columns(
                pl.when(pl.col("trade_date") == pl.col("ann_date"))
                .then(pl.col("_next_td"))
                .otherwise(pl.col("trade_date"))
                .alias("T_date")
            )
            .select(["ts_code", "ann_date", "T_date"])
            .drop_nulls(["T_date"])
        )
        event_T = tmp.filter(pl.col("T_date") > pl.col("ann_date"))

    # 标记行情表中哪些 (ts_code, trade_date) 是“财报反应日 T”
    event_marker = (
        event_T.select(["ts_code", pl.col("T_date").alias("trade_date")])
        .with_columns(pl.lit(1).cast(pl.Int8).alias("_is_T"))
    )
    df = df_price.sort(["ts_code", "trade_date"]).join(
        event_marker, on=["ts_code", "trade_date"], how="left"
    )
    is_event_day = pl.col("_is_T").is_not_null()

    # -------------------------------------------------------------------------
    # 2) 截面与时序特征（.over("ts_code")，防未来：shift(1) 或滞后）
    # -------------------------------------------------------------------------
    prev_close = pl.col("close").shift(1).over("ts_code")
    daily_ret = (pl.col("close") / prev_close - 1.0).alias("daily_ret")

    vol_ma20 = (
        pl.col("vol").rolling_mean(vol_ma_w).shift(1).over("ts_code").alias("vol_ma20")
    )
    close_1 = pl.col("close").shift(1).over("ts_code")
    close_21 = pl.col("close").shift(1 + pre_w).over("ts_code")
    pre_20d_ret = (close_1 / close_21 - 1.0).alias("pre_20d_ret")

    # 行号（用于持仓天数）：按 ts_code 组内顺序
    rn = pl.col("trade_date").rank("ordinal").over("ts_code").cast(pl.UInt32).alias("rn")

    df = df.with_columns([daily_ret, vol_ma20, pre_20d_ret, rn])
    vol_ratio = (pl.col("vol") / pl.col("vol_ma20")).alias("vol_ratio")
    df = df.with_columns([vol_ratio])

    # -------------------------------------------------------------------------
    # 3) 信号触发：仅在财报反应日 T，满足全部条件则 entry_signal = 1
    # -------------------------------------------------------------------------
    entry_cond = (
        is_event_day
        & (pl.col("daily_ret") > daily_ret_th)
        & (pl.col("vol_ratio") > vol_ratio_th)
        & (pl.col("pre_20d_ret") < pre_ret_cap)
    )
    entry_signal = entry_cond.fill_null(False).cast(pl.Int8).alias("entry_signal")

    df = df.with_columns([entry_signal])

    # -------------------------------------------------------------------------
    # 4) 持仓状态机：按“段”维护 entry_open / entry_rn，再算退出
    #    段 = 每次 entry_signal=1 开始的新区间（segment_id = entry_signal.cum_sum）
    # -------------------------------------------------------------------------
    segment_id = pl.col("entry_signal").cum_sum().over("ts_code").alias("segment_id")
    df = df.with_columns([segment_id])

    entry_open = (
        pl.when(pl.col("entry_signal") == 1)
        .then(pl.col("open"))
        .otherwise(None)
        .forward_fill()
        .over(["ts_code", "segment_id"])
        .alias("entry_open")
    )
    entry_rn = (
        pl.when(pl.col("entry_signal") == 1)
        .then(pl.col("rn"))
        .otherwise(None)
        .forward_fill()
        .over(["ts_code", "segment_id"])
        .alias("entry_rn")
    )
    df = df.with_columns([entry_open, entry_rn])

    days_held = (pl.col("rn") - pl.col("entry_rn")).alias("days_held")
    df = df.with_columns([days_held])

    # 退出：持仓满 hold_days 或 收盘价跌破 T 日开盘价（且已持仓至少 1 日）
    exit_cond = (
        (pl.col("segment_id") > 0)
        & (
            (pl.col("days_held") >= hold_days)
            | ((pl.col("days_held") > 0) & (pl.col("close") < pl.col("entry_open")))
        )
    )
    exit_signal = exit_cond.fill_null(False).cast(pl.Int8).alias("exit_signal")
    df = df.with_columns([exit_signal])

    # 持仓状态：入场=1，出场=0，否则延续（forward_fill）
    raw_state = (
        pl.when(pl.col("entry_signal") == 1)
        .then(1)
        .when(pl.col("exit_signal") == 1)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
    )
    position_status = raw_state.forward_fill().over("ts_code").fill_null(0).alias("position_status")
    df = df.with_columns([position_status])

    # 事件日标记（便于回测/分析）
    is_event_day_col = is_event_day.alias("is_event_day")
    df = df.with_columns([is_event_day_col])

    # 清理中间列，保留与策略/回测相关的输出
    drop_cols = ["_is_T", "rn", "segment_id", "entry_rn", "days_held"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    out = df.drop(existing_drop) if existing_drop else df

    return out


def export_pead_signal_file(
    df_signal: pl.DataFrame,
    *,
    out_path: str = "SM-策略-09-pead_signals.parquet",
) -> str:
    """
    将策略输出的信号表“对齐回测口径”并落地为文件。

    回测所需列：
    - ts_code, trade_date, open, close, entry_signal
    - event_open: 仅在 entry_signal==1 的当日记录 open（止损线）
    - daily_ret: close/close.shift(1)-1（若已有则沿用）
    """
    if "daily_ret" not in df_signal.columns:
        df_signal = df_signal.sort(["ts_code", "trade_date"]).with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("daily_ret")
        )

    df_signal = df_signal.with_columns(
        pl.when(pl.col("entry_signal") == 1).then(pl.col("open")).otherwise(None).alias("event_open")
    )

    # 标记价格口径（便于回测端做校验/追溯）
    df_signal = df_signal.with_columns(
        pl.lit("pre_adjust_to_last").alias("adj_method"),
        pl.lit(1).cast(pl.Int8).alias("is_adj_price"),
    )

    keep = [
        "ts_code",
        "trade_date",
        "open",
        "close",
        "daily_ret",
        "entry_signal",
        "event_open",
        "is_adj_price",
        "adj_method",
    ]
    df_out = df_signal.select([c for c in keep if c in df_signal.columns]).sort(["trade_date", "ts_code"])

    abs_path = out_path
    if not os.path.isabs(abs_path):
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    df_out.write_parquet(abs_path)
    return abs_path


def load_pead_inputs_from_duckdb(
    db_path: str,
    *,
    start_date: str = "20150101",
    end_date: str | None = None,
    ann_table: str = "fina_indicator",
    ann_date_col: str = "ann_date",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    从 DuckDB 读取 PEAD 所需输入：
    - df_price: ts_code, trade_date, open, close, vol（默认使用前复权 open/close）
    - df_ann: ts_code, ann_date
    """
    import duckdb

    end = end_date or datetime.now().strftime("%Y%m%d")
    con = duckdb.connect(db_path, read_only=True)
    try:
        d = con.execute(
            "SELECT ts_code, trade_date, open, close, vol FROM daily_data WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
        a = con.execute(
            "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
        ann = con.execute(
            f"SELECT ts_code, {ann_date_col} AS ann_date FROM {ann_table} WHERE {ann_date_col} >= ? AND {ann_date_col} <= ?",
            [start_date, end],
        ).pl()
    finally:
        con.close()

    j = d.join(a, on=["ts_code", "trade_date"], how="inner").sort(["ts_code", "trade_date"])
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af
    price = j.with_columns(
        (pl.col("open") * ratio).alias("open"),
        (pl.col("close") * ratio).alias("close"),
    ).select(["ts_code", "trade_date", "open", "close", "vol"])

    return price, ann.select(["ts_code", "ann_date"]).unique()


if __name__ == "__main__":
    # 生成“回测口径信号文件”：优先 DuckDB，否则跑最小示例
    db_path = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)

    if os.path.isfile(db_path):
        try:
            df_price, df_ann = load_pead_inputs_from_duckdb(
                db_path,
                start_date=os.getenv("PEAD_START_DATE", "20150101"),
                end_date=os.getenv("PEAD_END_DATE") or None,
                ann_table=os.getenv("PEAD_ANN_TABLE", "fina_indicator"),
                ann_date_col=os.getenv("PEAD_ANN_DATE_COL", "ann_date"),
            )
            out = generate_pead_signals(df_price, df_ann)
            path = export_pead_signal_file(out, out_path=os.getenv("PEAD_SIGNAL_OUT", "pead_signals.parquet"))
            print(f"已保存 PEAD 信号文件: {path}，行数={out.height:,}，股票数={out['ts_code'].n_unique():,}")
        except Exception as e:
            print(f"从 DuckDB 生成并导出 PEAD 信号失败：{e}")
    else:
        # 最小示例：构造行情 + 公告，验证事件日与信号
        import random
        random.seed(42)
        codes = ["000001.SZ", "000002.SZ"]
        dates = [f"2024{i//30+1:02d}{(i%30)+1:02d}" for i in range(90)]
        rows = []
        for c in codes:
            base = 10.0
            for i, d in enumerate(dates):
                ret = 0.001 * (random.random() - 0.5)
                base = base * (1 + ret)
                vol = 1e6 * (1 + random.random())
                rows.append({"ts_code": c, "trade_date": d, "open": base * 0.99, "close": base, "vol": vol})
        df_price = pl.DataFrame(rows)
        df_ann = pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "ann_date": ["20240114", "20240120"]})
        out = generate_pead_signals(df_price, df_ann)
        path = export_pead_signal_file(out, out_path="pead_signals.parquet")
        print(f"已保存 PEAD 信号文件: {path}")
