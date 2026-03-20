"""
动态配对挖掘机 · Cointegration Pair Finder（Monthly Rebalance）

目标：
- 从 DuckDB 中按月动态挖掘全市场最优的 100 对协整股票
- Formation Period：过去 250 个交易日（约 1 年）
- 三级降维漏斗：
  1) Top300 高流动性股票池（算力保护）
  2) Pearson 相关系数过滤（corr>=0.85）
  3) Engle-Granger 协整检验（coint p_value<0.05）

输出：
- 追加保存到本地表：shiming_dynamic_pairs.parquet
  字段：month, stock_A, stock_B, p_value

性能：
- 协整检验使用多进程并行（ProcessPoolExecutor）
- tqdm 显示“月度扫描进度 + 当月协整检验进度”
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass(frozen=True)
class PairFinderConfig:
    duckdb_path: str = "shiming_daily_base.duckdb"
    out_parquet: str = "shiming_dynamic_pairs.parquet"

    formation_window: int = 250
    top_n_liquid: int = 300

    corr_th: float = 0.85
    coint_p_th: float = 0.05
    top_k_pairs: int = 100

    # 流动性打分：优先 amount（成交额），否则 total_mv（市值）
    # 使用 formation 窗口内的均值作为“月末可用信息”的稳健近似
    liquidity_metric: str = "amount"  # amount | total_mv

    # 并行
    max_workers: int | None = None  # None -> os.cpu_count()


def _ensure_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _read_from_duckdb(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pl.DataFrame:
    try:
        if params:
            return con.execute(sql, params).pl()
        return con.execute(sql).pl()
    except Exception:
        # 兼容 duckdb < 0.9/部分环境：回退 arrow
        if params:
            return pl.from_arrow(con.execute(sql, params).arrow())
        return pl.from_arrow(con.execute(sql).arrow())


def _month_ends(trade_dates: list[str]) -> list[str]:
    """
    输入已排序的 YYYYMMDD 字符串列表，返回每月最后一个交易日（同口径月末）。
    """
    if not trade_dates:
        return []
    td = pd.Series(trade_dates)
    m = td.str.slice(0, 6)
    # 每个 month 的最后一行
    idx = m.ne(m.shift(-1))
    return td[idx].tolist()


def _pick_top_liquid(
    df: pl.DataFrame,
    top_n: int,
    *,
    metric: str,
) -> list[str]:
    """
    df: formation window 内数据（ts_code, trade_date, close, amount?, total_mv?）
    返回 top_n 的 ts_code 列表。
    """
    cols = set(df.columns)
    use_metric = None
    if metric == "amount" and "amount" in cols:
        use_metric = "amount"
    elif "total_mv" in cols:
        use_metric = "total_mv"
    elif "amount" in cols:
        use_metric = "amount"
    else:
        raise KeyError("缺少流动性字段：需要 amount 或 total_mv")

    scored = (
        df.group_by("ts_code")
        .agg(pl.col(use_metric).mean().alias("_liq"))
        .sort("_liq", descending=True)
        .head(int(top_n))
        .select("ts_code")
    )
    return scored.get_column("ts_code").to_list()


def _build_wide_close_panel(df: pl.DataFrame, codes: list[str]) -> pd.DataFrame:
    """
    输入长表：trade_date, ts_code, close
    输出宽表：index=trade_date（datetime64[ns]），columns=ts_code，values=close
    """
    x = (
        df.filter(pl.col("ts_code").is_in(codes))
        .select(["trade_date", "ts_code", "close"])
        .with_columns(
            [
                pl.col("trade_date").cast(pl.Utf8),
                pl.col("ts_code").cast(pl.Utf8),
                pl.col("close").cast(pl.Float64).fill_nan(None),
            ]
        )
    )

    # Polars pivot → Pandas（更快 corr）
    wide = x.pivot(values="close", index="trade_date", on="ts_code", aggregate_function="first")
    pdf = wide.to_pandas()

    # 规范 trade_date
    pdf.index = pd.to_datetime(pdf.index.astype(str), format="%Y%m%d", errors="coerce")
    pdf = pdf.sort_index()

    # 强制数值化（避免 object/str 导致 numpy ufunc 报错）
    pdf = pdf.apply(pd.to_numeric, errors="coerce")

    # 严格：删除全空列
    pdf = pdf.dropna(axis=1, how="all")
    return pdf


def _corr_candidate_pairs(close_panel: pd.DataFrame, corr_th: float) -> list[tuple[str, str, float]]:
    """
    返回候选 pair 列表 (A, B, corr)，只取上三角且 corr>=阈值。
    """
    # 用 log price 更稳健；遇到非正价格则自动 NaN
    logp = np.log(close_panel.replace({0.0: np.nan}))
    corr = logp.corr(method="pearson", min_periods=max(60, int(len(logp) * 0.6)))

    cols = corr.columns.to_list()
    mat = corr.to_numpy()

    out: list[tuple[str, str, float]] = []
    n = len(cols)
    for i in range(n):
        # 只取上三角
        row = mat[i, i + 1 :]
        if row.size == 0:
            continue
        js = np.where(row >= corr_th)[0]
        if js.size == 0:
            continue
        a = cols[i]
        for j0 in js.tolist():
            j = i + 1 + int(j0)
            b = cols[j]
            out.append((a, b, float(mat[i, j])))
    return out


def _coint_worker(args: tuple[str, str, np.ndarray, np.ndarray]) -> tuple[str, str, float]:
    a, b, s_a, s_b = args
    # 清理 nan
    mask = np.isfinite(s_a) & np.isfinite(s_b)
    s_a = s_a[mask]
    s_b = s_b[mask]
    # 样本太短直接丢弃
    if s_a.size < 120:
        return a, b, float("nan")
    try:
        _t, p, _crit = coint(s_a, s_b)
        return a, b, float(p)
    except Exception:
        return a, b, float("nan")


def _run_coint_parallel(
    close_panel: pd.DataFrame,
    pairs: list[tuple[str, str, float]],
    *,
    max_workers: int | None,
    desc: str,
) -> list[tuple[str, str, float]]:
    """
    对候选 pairs 并行执行 coint，返回 (A, B, p_value)。
    """
    if not pairs:
        return []

    # 预提取为 numpy，加速 worker 访问（避免在子进程里做 pandas slice 太慢）
    values = close_panel.to_numpy(dtype="float64", copy=False)
    col_index = {c: i for i, c in enumerate(close_panel.columns.to_list())}

    tasks = []
    for a, b, _corr in pairs:
        ia = col_index.get(a)
        ib = col_index.get(b)
        if ia is None or ib is None:
            continue
        tasks.append((a, b, values[:, ia], values[:, ib]))

    out: list[tuple[str, str, float]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_coint_worker, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc, leave=False):
            a, b, p = f.result()
            out.append((a, b, p))
    return out


def find_dynamic_pairs(
    config: PairFinderConfig | None = None,
    *,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    主函数：按月输出 top100 协整 pair，并落地 parquet（追加写）。
    返回本次运行新增的结果（DataFrame）。
    """
    cfg = config or PairFinderConfig()
    db_path = _ensure_abs(cfg.duckdb_path)
    out_path = _ensure_abs(cfg.out_parquet)
    end = end_date or datetime.now().strftime("%Y%m%d")

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"未找到 DuckDB：{db_path}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        # 拉取全市场 close + amount；并尽量带 total_mv
        df = _read_from_duckdb(
            con,
            """
            SELECT d.ts_code, d.trade_date, d.close, d.amount, b.total_mv
            FROM daily_data d
            LEFT JOIN daily_basic b
            ON d.ts_code=b.ts_code AND d.trade_date=b.trade_date
            WHERE d.trade_date >= ? AND d.trade_date <= ?
            """,
            [start_date, end],
        )
    finally:
        con.close()

    if df.is_empty():
        raise ValueError("DuckDB 查询结果为空：请检查日期区间或数据是否入库")

    # 清洗 + 排序
    df = (
        df.with_columns(
            [
                pl.col("ts_code").cast(pl.Utf8),
                pl.col("trade_date").cast(pl.Utf8),
                pl.col("close").cast(pl.Float64).fill_nan(None),
                pl.col("amount").cast(pl.Float64).fill_nan(None),
                pl.col("total_mv").cast(pl.Float64).fill_nan(None),
            ]
        )
        .drop_nulls(["ts_code", "trade_date", "close"])
        .sort(["trade_date", "ts_code"])
    )

    trade_dates = df.get_column("trade_date").unique().sort().to_list()
    month_ends = _month_ends(trade_dates)

    added_rows: list[dict] = []
    # 月度扫描：只处理“有足够 formation_window 历史”的月末
    td_pos = {d: i for i, d in enumerate(trade_dates)}

    months_iter = []
    for me in month_ends:
        i = td_pos.get(me)
        if i is None or i + 1 < cfg.formation_window:
            continue
        months_iter.append(me)

    for me in tqdm(months_iter, desc="Monthly scan", leave=True):
        i = td_pos[me]
        window_dates = trade_dates[i + 1 - cfg.formation_window : i + 1]

        # formation window 子集
        w = df.filter(pl.col("trade_date").is_in(window_dates))

        # 1) Top300 流动性池
        top_codes = _pick_top_liquid(w, cfg.top_n_liquid, metric=cfg.liquidity_metric)
        w = w.filter(pl.col("ts_code").is_in(top_codes))

        # 2) Pivot 宽表 close_panel
        panel = _build_wide_close_panel(w.rename({"close": "close"}), top_codes)

        # 对齐：剔除缺失过多的列（保留至少 80% 有效样本）
        min_obs = int(cfg.formation_window * 0.8)
        panel = panel.loc[:, panel.notna().sum(axis=0) >= min_obs]
        if panel.shape[1] < 2:
            continue

        # 3) 漏斗1：相关性过滤
        candidates = _corr_candidate_pairs(panel, cfg.corr_th)
        if not candidates:
            continue

        # 4) 漏斗2：并行协整检验
        coint_res = _run_coint_parallel(
            panel,
            candidates,
            max_workers=cfg.max_workers,
            desc=f"coint {me}",
        )

        # 5) 优中选优：p_value < 0.05 → 最小的前 100
        valid = [(a, b, p) for a, b, p in coint_res if np.isfinite(p) and p < cfg.coint_p_th]
        if not valid:
            continue

        valid.sort(key=lambda x: x[2])
        topk = valid[: int(cfg.top_k_pairs)]

        month = str(me)[:6]
        for a, b, p in topk:
            # 规范 pair 顺序，去重更稳
            stock_a, stock_b = (a, b) if a <= b else (b, a)
            added_rows.append(
                {
                    "month": month,
                    "stock_A": stock_a,
                    "stock_B": stock_b,
                    "p_value": float(p),
                }
            )

    added = pl.DataFrame(added_rows) if added_rows else pl.DataFrame(
        {"month": [], "stock_A": [], "stock_B": [], "p_value": []},
        schema={"month": pl.Utf8, "stock_A": pl.Utf8, "stock_B": pl.Utf8, "p_value": pl.Float64},
    )

    # 追加保存到 parquet（parquet 无原生 append：读旧 + concat + 覆盖写）
    if not added.is_empty():
        if os.path.isfile(out_path):
            old = pl.read_parquet(out_path)
            merged = (
                pl.concat([old, added], how="vertical")
                .unique(subset=["month", "stock_A", "stock_B"], keep="last")
                .sort(["month", "p_value"])
            )
        else:
            merged = added.sort(["month", "p_value"])
        merged.write_parquet(out_path)

    return added


if __name__ == "__main__":
    cfg = PairFinderConfig(
        duckdb_path=os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb"),
        out_parquet="shiming_dynamic_pairs.parquet",
        formation_window=250,
        top_n_liquid=300,
        corr_th=0.85,
        coint_p_th=0.05,
        top_k_pairs=100,
        liquidity_metric="amount",
        max_workers=None,
    )

    added = find_dynamic_pairs(cfg, start_date=os.getenv("START_DATE", "20150101"), end_date=None)
    print("✅ 本次新增 pairs 行数:", added.height)
    if added.height > 0:
        print(added.head(10))
