"""
Dual Thrust · 回测框架（独立入口）

- 从 shiming_daily_base.duckdb 读取日线并做前复权（可直接喂给策略）
- 动态导入 SM-策略-02-dual thrust.py 的 generate_dual_thrust_signals
- 资金管理：Fixed Slot Allocation（固定槽位等额分配），支持同日多信号竞争排序
- 结算：账户模拟法（cash + 持仓市值），输出日度指标与净值曲线
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
    name = "SM-策略-02-dual thrust.py"
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到策略文件：{name}")

    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("_dual_thrust", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "generate_dual_thrust_signals")


generate_dual_thrust_signals = _load_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20210101",
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
    # 前复权：每只股用该股最新交易日的 adj_factor 做分母
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
    """Dual Thrust 回测参数（周期、资金、槽位、费用、排序规则）"""

    start_date: str = "20210101"
    end_date: str | None = None

    initial_capital: float = 10_000_000
    n_slots: int = 10  # 固定槽位数量
    slot_cash: float | None = None  # None 表示 initial_capital / n_slots

    # 成本：买入只扣佣金滑点；卖出扣佣金+印花税（示例用你给的 0.0013）
    buy_cost: float = 0.0003
    sell_cost: float = 0.0013

    # 信号执行：默认 T+1（信号日收盘确认，次日开盘成交）
    t_plus_one: bool = True

    # 同日多信号竞争排序（可选）：strength / ret / none
    rank_mode: str = "strength"

    # Dual Thrust 参数（透传给策略）
    N: int = 5
    K1: float = 0.5
    K2: float = 0.5


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
    Dual Thrust 回测（Fixed Slot Allocation）。

    资金分配核心：
    - 槽位资金 slot_cash = initial_capital / n_slots（或用户指定）
    - 满足 entry_signal 且槽位未满时：shares = floor(slot_cash / open / 100) * 100
      并从现金扣除：shares*open*(1+buy_cost)
    - 触发 exit_signal：卖出全部 shares，现金回笼：shares*open*(1-sell_cost)
      注意：卖出释放的槽位 **T+1 才能复用** → 当天先统计可用槽位=开盘前的空槽位数，卖出不增加当日买入额度。

    返回：
    - daily：按 trade_date 的日度净值与监控指标
    - meta：包含 missed_signals_due_to_slots 等统计
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")
    slot_cash = float(cfg.slot_cash) if cfg.slot_cash is not None else float(cfg.initial_capital) / float(cfg.n_slots)

    df = _unify_prices(df)
    df = df.filter((pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end))
    df = df.sort(["ts_code", "trade_date"])

    # 1) 生成策略信号（纯向量化）
    df = generate_dual_thrust_signals(df, N=cfg.N, K1=cfg.K1, K2=cfg.K2)

    # 2) 信号执行：T+1（默认）在次日开盘成交
    if cfg.t_plus_one:
        df = df.with_columns(
            pl.col("entry_signal").shift(1).over("ts_code").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").shift(1).over("ts_code").fill_null(0).alias("actual_exit"),
            # 用信号日的突破强度做排序（shift 到执行日）
            ((pl.col("close") - pl.col("buy_line")) / pl.col("buy_line"))
            .replace([float("inf"), float("-inf")], None)
            .fill_nan(None)
            .shift(1)
            .over("ts_code")
            .alias("actual_strength"),
            ((pl.col("close") / pl.col("open")) - 1.0)
            .replace([float("inf"), float("-inf")], None)
            .fill_nan(0.0)
            .shift(1)
            .over("ts_code")
            .alias("actual_intraday_ret"),
        )
    else:
        df = df.with_columns(
            pl.col("entry_signal").fill_null(0).alias("actual_entry"),
            pl.col("exit_signal").fill_null(0).alias("actual_exit"),
            ((pl.col("close") - pl.col("buy_line")) / pl.col("buy_line"))
            .replace([float("inf"), float("-inf")], None)
            .fill_nan(None)
            .alias("actual_strength"),
            ((pl.col("close") / pl.col("open")) - 1.0)
            .replace([float("inf"), float("-inf")], None)
            .fill_nan(0.0)
            .alias("actual_intraday_ret"),
        )

    # 3) 账户模拟：按交易日 group_by（保持时间顺序）
    cash = float(cfg.initial_capital)
    holdings: dict[str, int] = {}  # ts_code -> shares（一个标的占用一个槽位）
    last_close: dict[str, float] = {}

    missed_due_to_slots = 0
    missed_due_to_cash = 0
    executed_buys = 0
    executed_sells = 0

    daily_records: list[dict] = []

    needed = [
        "ts_code",
        "open",
        "close",
        "actual_entry",
        "actual_exit",
        "actual_strength",
        "actual_intraday_ret",
    ]

    for (trade_date,), day in df.group_by("trade_date", maintain_order=True):
        rows = list(day.select(needed).iter_rows(named=True))

        # 当日开盘前总资产：cash + 持仓市值（昨日收盘价估值）
        equity_start = cash
        for code, sh in holdings.items():
            if sh <= 0:
                continue
            equity_start += sh * last_close.get(code, 0.0)

        # 可用槽位（卖出释放 T+1 才可复用，因此按开盘前的持仓数算）
        occupied_start = sum(1 for _c, sh in holdings.items() if sh > 0)
        slots_left_today = max(0, int(cfg.n_slots) - occupied_start)

        # --------------------------
        # A) 先处理卖出（当日开盘成交，释放槽位但不增加当天 slots_left_today）
        # --------------------------
        for r in rows:
            code = r["ts_code"]
            if int(r["actual_exit"]) != 1:
                continue
            sh = holdings.get(code, 0)
            if sh <= 0:
                continue
            sell_amt = sh * float(r["open"])
            cash += sell_amt * (1.0 - float(cfg.sell_cost))
            holdings.pop(code, None)
            executed_sells += 1

        # --------------------------
        # B) 再处理买入（当日开盘成交；同日多信号按排序优先填满槽位）
        # --------------------------
        candidates = []
        for r in rows:
            code = r["ts_code"]
            if int(r["actual_entry"]) != 1:
                continue
            if holdings.get(code, 0) > 0:
                continue
            candidates.append(r)

        if candidates:
            if cfg.rank_mode == "strength":
                candidates.sort(
                    key=lambda x: (x["actual_strength"] if x["actual_strength"] is not None else float("-inf")),
                    reverse=True,
                )
            elif cfg.rank_mode == "ret":
                candidates.sort(
                    key=lambda x: (x["actual_intraday_ret"] if x["actual_intraday_ret"] is not None else float("-inf")),
                    reverse=True,
                )
            else:
                # none：保持 ts_code 字典序（稳定、可复现）
                candidates.sort(key=lambda x: x["ts_code"])

            if len(candidates) > slots_left_today:
                missed_due_to_slots += len(candidates) - slots_left_today
                candidates = candidates[:slots_left_today]

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

        # 更新收盘价缓存（用于次日估值）
        for r in rows:
            last_close[r["ts_code"]] = float(r["close"])

        # 当日收盘总资产：cash + 持仓市值（今日收盘价；若缺失则回退 last_close）
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
                "slots_left_today": slots_left_today,
            }
        )

    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        pl.col("strategy_return").fill_nan(0.0).clip(-0.99, 10.0),
        (1.0 + pl.col("strategy_return")).cum_prod().alias("cum_nav"),
    )

    meta = {
        "missed_due_to_slots": missed_due_to_slots,
        "missed_due_to_cash": missed_due_to_cash,
        "executed_buys": executed_buys,
        "executed_sells": executed_sells,
        "slot_cash": slot_cash,
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

    print("=== Dual Thrust 回测结果（Fixed Slots）===")
    print(f"区间:         {daily['trade_date'][0]} ~ {daily['trade_date'][-1]}")
    print(f"总收益率:     {total_return:.4f}")
    print(f"年化收益:     {ann_return:.4f}")
    print(f"最大回撤:     {max_dd:.4f}")
    print(f"Calmar 比率:  {calmar:.4f}")
    print(f"胜率(日):     {win_rate:.4f}")
    print("--- 资金与槽位 ---")
    print(f"初始资金:     {initial_capital:,.0f}")
    print(f"槽位数:       {meta.get('n_slots', 'N/A')}")
    print(f"单槽资金:     {meta.get('slot_cash', 0):,.0f}")
    print(f"执行买入:     {meta.get('executed_buys', 0)}")
    print(f"执行卖出:     {meta.get('executed_sells', 0)}")
    print(f"槽位不足错过: {meta.get('missed_due_to_slots', 0)}")
    print(f"现金不足错过: {meta.get('missed_due_to_cash', 0)}")


def plot_nav(daily: pl.DataFrame, path: str = "dual_thrust_nav.png") -> None:
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

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(d["trade_date"], d["cum_nav"], lw=1.5, label="NAV")
    ax1.set_title("Dual Thrust Strategy NAV (Fixed Slots)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("NAV")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已保存: {path}")


if __name__ == "__main__":
    config = BacktestConfig(
        start_date="20210101",
        end_date=None,
        initial_capital=10_000_000,
        n_slots=10,
        slot_cash=None,
        buy_cost=0.0003,
        sell_cost=0.0013,
        t_plus_one=True,
        rank_mode="strength",
        N=5,
        K1=0.5,
        K2=0.5,
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
    # 补充 meta 信息用于展示
    meta = {**meta, "n_slots": config.n_slots}
    print_stats(daily, meta, initial_capital=config.initial_capital)
    plot_nav(daily)

