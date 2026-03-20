"""
BOLL 极限反转 · 回测框架（独立入口）

- 动态导入 SM-策略-03-boll.py 的 generate_boll_reversal_signals
- T+1 执行：信号日(T)收盘确认，T+1 以开盘价(open)成交
- 资金管理：Fixed Slot Allocation（10 个槽位，每槽 100 万，最多持仓 10 只）
- 多股并发“选美”排序：按 T 日“超跌偏离度” = (close - lower_band) / lower_band
  偏离度越负（跌得越透），优先级越高（即排序升序）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import duckdb
import polars as pl


# A 股最小交易单位（手）
LOT_SIZE = 100


def _load_strategy():
    root = os.path.dirname(os.path.abspath(__file__))
    name = "SM-策略-03-boll.py"
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到策略文件：{name}")

    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("_boll", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "generate_boll_reversal_signals")


generate_boll_reversal_signals = _load_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    从 shiming_daily_base.duckdb 读取 daily_data + adj_factor，计算前复权 OHLC，返回 DataFrame。
    返回列：ts_code, trade_date, adj_open, adj_high, adj_low, adj_close（回测内会统一为 open/high/low/close）。
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    con = duckdb.connect(db_path, read_only=True)
    try:
        d = con.execute(
            "SELECT ts_code, trade_date, open, high, low, close FROM daily_data WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
        a = con.execute(
            "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date >= ? AND trade_date <= ?",
            [start_date, end],
        ).pl()
    finally:
        con.close()

    j = d.join(a, on=["ts_code", "trade_date"], how="inner").sort(["ts_code", "trade_date"])
    last_af = pl.col("adj_factor").last().over("ts_code")
    ratio = pl.col("adj_factor") / last_af
    return j.with_columns(
        (pl.col("open") * ratio).alias("adj_open"),
        (pl.col("high") * ratio).alias("adj_high"),
        (pl.col("low") * ratio).alias("adj_low"),
        (pl.col("close") * ratio).alias("adj_close"),
    ).select(["ts_code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close"])


@dataclass
class BacktestConfig:
    """BOLL 回测参数（周期、资金、槽位、费用）"""

    start_date: str = "20150101"
    end_date: str | None = None

    initial_capital: float = 10_000_000
    n_slots: int = 10
    slot_cash: float | None = None  # None 表示 initial_capital / n_slots

    buy_cost: float = 0.0003   # 买入万3
    sell_cost: float = 0.0013  # 卖出千1.3（含印花税+佣金）

    t_plus_one: bool = True


def _unify_prices(df: pl.DataFrame) -> pl.DataFrame:
    """统一价格列名（优先使用复权价）。"""
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))
    if "adj_open" in df.columns and "open" not in df.columns:
        df = df.with_columns(pl.col("adj_open").alias("open"))
    if "adj_high" in df.columns and "high" not in df.columns:
        df = df.with_columns(pl.col("adj_high").alias("high"))
    if "adj_low" in df.columns and "low" not in df.columns:
        df = df.with_columns(pl.col("adj_low").alias("low"))
    return df


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, dict]:
    """
    BOLL 极限反转回测（Fixed Slot + T+1 + 选美排序）。

    交易规则（严格按需求）：
    - 若 T 日 entry_signal==1，则 T+1 以 open 买入
    - 若 T 日 exit_signal==1，则 T+1 以 open 全部卖出
    - 槽位：10 个等额槽位（每槽 slot_cash），最多同时持仓 10 只
    - 同日多只入场：按 T 日“超跌偏离度”=(close-lower_band)/lower_band 排序，越负越优先
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")
    slot_cash = float(cfg.slot_cash) if cfg.slot_cash is not None else float(cfg.initial_capital) / float(cfg.n_slots)

    df = _unify_prices(df)
    df = df.filter((pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end))
    df = df.sort(["ts_code", "trade_date"])

    # 1) 生成策略信号（纯向量化）
    df = generate_boll_reversal_signals(df)

    # 2) 计算超跌偏离度（信号日 T）：(close - lower_band) / lower_band
    #    并 shift(1) 到执行日（T+1）用于当日“选美”排序
    dev = ((pl.col("close") - pl.col("lower_band")) / pl.col("lower_band")).replace(
        [float("inf"), float("-inf")], None
    ).fill_nan(None)

    if cfg.t_plus_one:
        df = df.with_columns(
            pl.col("entry_signal").shift(1).over("ts_code").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").shift(1).over("ts_code").fill_null(0).alias("actual_exit"),
            dev.shift(1).over("ts_code").alias("actual_dev"),
        )
    else:
        df = df.with_columns(
            pl.col("entry_signal").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").fill_null(0).alias("actual_exit"),
            dev.alias("actual_dev"),
        )

    # 3) 账户模拟：按日迭代（槽位模型需要动态现金/持仓）
    cash = float(cfg.initial_capital)
    holdings: dict[str, int] = {}   # ts_code -> shares（一个标的占一个槽位）
    last_close: dict[str, float] = {}

    missed_due_to_slots = 0
    missed_due_to_cash = 0
    executed_buys = 0
    executed_sells = 0

    daily_records: list[dict] = []

    needed = ["ts_code", "open", "close", "actual_entry", "actual_exit", "actual_dev"]

    for (trade_date,), day in df.group_by("trade_date", maintain_order=True):
        rows = list(day.select(needed).iter_rows(named=True))

        # 开盘前总资产：现金 + 持仓市值（按昨收估值）
        equity_start = cash
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            equity_start += sh * last_close.get(code, 0.0)

        # A) 先卖出（T 日出场 → T+1 开盘卖出），释放槽位与现金（可用于当日买入）
        for r in rows:
            if int(r["actual_exit"]) != 1:
                continue
            code = r["ts_code"]
            sh = holdings.get(code, 0)
            if sh <= 0:
                continue
            sell_amt = sh * float(r["open"])
            cash += sell_amt * (1.0 - float(cfg.sell_cost))
            holdings.pop(code, None)
            executed_sells += 1

        # 计算当前空闲槽位（卖出已发生，槽位可当日复用）
        occupied = sum(1 for _c, sh in holdings.items() if sh > 0)
        slots_left = max(0, int(cfg.n_slots) - occupied)

        # B) 收集当日买入候选（T 日入场 → T+1 开盘买入），按超跌偏离度排序（越负越优）
        candidates = []
        for r in rows:
            if int(r["actual_entry"]) != 1:
                continue
            code = r["ts_code"]
            if holdings.get(code, 0) > 0:
                continue
            candidates.append(r)

        if candidates:
            # 偏离度越负越优先；None 放到队尾
            candidates.sort(
                key=lambda x: (x["actual_dev"] if x["actual_dev"] is not None else float("inf"))
            )

            if len(candidates) > slots_left:
                missed_due_to_slots += len(candidates) - slots_left
                candidates = candidates[:slots_left]

            for r in candidates:
                price = float(r["open"])
                if price <= 0:
                    continue
                shares = int(slot_cash / price / LOT_SIZE) * LOT_SIZE
                if shares < LOT_SIZE:
                    missed_due_to_cash += 1
                    continue
                need_cash = shares * price * (1.0 + float(cfg.buy_cost))
                if need_cash > cash:
                    missed_due_to_cash += 1
                    continue
                cash -= need_cash
                holdings[r["ts_code"]] = shares
                executed_buys += 1

        # 更新收盘价缓存
        for r in rows:
            last_close[r["ts_code"]] = float(r["close"])

        # 收盘总资产：现金 + 持仓市值（按今收）
        equity_end = cash
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            equity_end += sh * last_close.get(code, 0.0)

        daily_pnl = equity_end - equity_start
        daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
        daily_ret = max(-0.99, min(10.0, float(daily_ret)))

        holding_cnt = sum(1 for _c, sh in holdings.items() if sh > 0)
        cash_usage = 1.0 - (cash / equity_end) if equity_end > 1e-8 else 0.0

        daily_records.append(
            {
                "trade_date": trade_date,
                "equity_start": equity_start,
                "equity_end": equity_end,
                "daily_pnl": daily_pnl,
                "strategy_return": daily_ret,
                "holdings": holding_cnt,
                "cash_usage": cash_usage,
                "slots_left": slots_left,
            }
        )

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        pl.col("strategy_return").fill_nan(0.0).clip(-0.99, 10.0),
        (1.0 + pl.col("strategy_return")).cum_prod().alias("cum_nav"),
    )

    meta = {
        "n_slots": cfg.n_slots,
        "slot_cash": slot_cash,
        "missed_due_to_slots": missed_due_to_slots,
        "missed_due_to_cash": missed_due_to_cash,
        "executed_buys": executed_buys,
        "executed_sells": executed_sells,
    }
    return daily, meta


def print_stats(daily: pl.DataFrame, meta: dict, initial_capital: float) -> None:
    if daily.is_empty():
        print("无交易日数据。")
        return
    nav = daily["cum_nav"].fill_nan(1.0).clip(1e-8, 1e10)
    total_return = float(nav[-1]) - 1.0
    peak = nav.cum_max()
    max_dd = float(((nav / peak) - 1.0).min())
    win_rate = float((daily["strategy_return"] > 0).mean())

    n_days = len(daily)
    years = n_days / 252.0 if n_days else 0.0
    ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    print("=== BOLL 极限反转回测结果（Fixed Slots + Ranking）===")
    print(f"区间:         {daily['trade_date'][0]} ~ {daily['trade_date'][-1]}")
    print(f"总收益率:     {total_return:.4f}")
    print(f"年化收益:     {ann_return:.4f}")
    print(f"最大回撤:     {max_dd:.4f}")
    print(f"Calmar 比率:  {calmar:.4f}")
    print(f"胜率(日):     {win_rate:.4f}")
    print("--- 槽位与错过信号 ---")
    print(f"槽位数:       {meta.get('n_slots', 'N/A')}")
    print(f"单槽资金:     {meta.get('slot_cash', 0):,.0f}")
    print(f"执行买入:     {meta.get('executed_buys', 0)}")
    print(f"执行卖出:     {meta.get('executed_sells', 0)}")
    print(f"槽位不足错过: {meta.get('missed_due_to_slots', 0)}")
    print(f"现金不足错过: {meta.get('missed_due_to_cash', 0)}")


def plot_nav(daily: pl.DataFrame, path: str = "boll_nav.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    d = daily.to_pandas()
    d["trade_date"] = d["trade_date"].astype(str)
    if d["trade_date"].str.len().max() == 8:
        d["trade_date"] = d["trade_date"].str[:4] + "-" + d["trade_date"].str[4:6] + "-" + d["trade_date"].str[6:8]
    d["trade_date"] = d["trade_date"].astype("datetime64[ns]")

    plt.figure(figsize=(12, 5))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5)
    plt.title("BOLL Reversal Strategy NAV (Fixed Slots + Ranking)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存: {path}")


if __name__ == "__main__":
    config = BacktestConfig(
        start_date="20150101",
        end_date=None,
        initial_capital=10_000_000,
        n_slots=10,
        slot_cash=None,
        buy_cost=0.0003,
        sell_cost=0.0013,
        t_plus_one=True,
    )

    db_path = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"未找到数据库：{db_path}")

    df = load_from_duckdb(db_path, start_date=config.start_date, end_date=config.end_date)
    if df.is_empty():
        raise ValueError("DuckDB 查询结果为空")

    print(f"已从 {db_path} 加载 {len(df):,} 条日线，{df['ts_code'].n_unique()} 只标的。")
    daily, meta = run_backtest(df, config)
    print_stats(daily, meta, initial_capital=config.initial_capital)
    plot_nav(daily)

