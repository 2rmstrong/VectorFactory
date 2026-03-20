"""
海龟策略 · 回测框架（独立入口）
导入 SM-策略-01-turtle.py 的 generate_turtle_signals，按可配置周期、资金、T+1、手续费与 1% 风险执行回测。
支持从 shiming_daily_base.duckdb 读取日线并做前复权后回测。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import duckdb
import polars as pl

# ----------------- 策略加载：从 SM-策略-01-turtle.py 导入 -----------------
def _load_strategy():
    root = os.path.dirname(os.path.abspath(__file__))
    # 兼容旧文件名：优先加载新文件名，找不到再回退
    for name in (
        "SM-策略-01-turtle.py",
        "SM-策略-01-turtle+ART-A.py",
        "SM-策略-01-turtle+ART+A股.py",
    ):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("_turtle", path)
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "generate_turtle_signals")
    raise FileNotFoundError("未找到策略文件：SM-策略-01-turtle.py")

generate_turtle_signals = _load_strategy()


def load_from_duckdb(
    db_path: str,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    从 shiming_daily_base.duckdb 读取 daily_data + adj_factor，计算前复权 OHLC，返回带 adj_close 等列的 DataFrame。
    返回列：ts_code, trade_date, adj_open, adj_high, adj_low, adj_close（供 run_backtest 直接使用）。
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


# A 股最小交易单位（手）
LOT_SIZE = 100


@dataclass
class BacktestConfig:
    """回测框架参数（周期、资金、T+1、手续费、单次风险）"""
    start_date: str = "20150101"           # 周期起点
    end_date: str | None = None            # 周期终点，None 表示今天
    initial_capital: float = 10_000_000   # 初始现金（默认 1000 万人民币）
    risk_pct: float = 0.01                # 单笔风险占当前总资产比例（1%，海龟风险对等）
    cost_roundtrip: float = 0.0003        # 单边手续费+滑点 万3（买卖各扣一次）
    t_plus_one: bool = True               # 信号延迟 T+1 执行


def run_backtest(df: pl.DataFrame, config: BacktestConfig | None = None) -> tuple[pl.DataFrame, list[float]]:
    """
    对日线 df 做海龟回测（账户模拟法），返回按 trade_date 聚合的日收益与累计净值。

    df 须含：ts_code, trade_date, open, high, low, close（或 adj_* 复权列）。
    资金管理：cash + hold_shares，建仓日按「当前总资产 × risk_pct」换算风险金额再除以 ATR 得目标股数，
    向下取整 100 股，且受现金约束；仅在持股数变化日扣万三摩擦成本。
    返回：(daily DataFrame, trade_pnls 每笔平仓盈亏列表)。
    """
    cfg = config or BacktestConfig()
    end = cfg.end_date or datetime.now().strftime("%Y%m%d")

    # 统一价格列名（优先复权价）
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.with_columns(pl.col("adj_close").alias("close"))
    if "adj_open" in df.columns and "open" not in df.columns:
        df = df.with_columns(pl.col("adj_open").alias("open"))
    if "adj_high" in df.columns and "high" not in df.columns:
        df = df.with_columns(pl.col("adj_high").alias("high"))
    if "adj_low" in df.columns and "low" not in df.columns:
        df = df.with_columns(pl.col("adj_low").alias("low"))

    df = df.filter(
        (pl.col("trade_date") >= cfg.start_date) & (pl.col("trade_date") <= end)
    )

    # 1) 策略信号
    df = generate_turtle_signals(df)

    # 2) T+1：今日信号明日执行
    if cfg.t_plus_one:
        df = df.with_columns(
            pl.col("position_status").shift(1).over("ts_code").alias("actual_position"),
            pl.col("atr").shift(1).over("ts_code").alias("actual_atr"),
        )
    else:
        df = df.with_columns(
            pl.col("position_status").alias("actual_position"),
            pl.col("atr").alias("actual_atr"),
        )

    # 3) 辅助列：前一日仓位、前收、建仓日/出场日；ATR 下限防除零
    df = df.with_columns(
        pl.col("actual_position").shift(1).over("ts_code").alias("prev_position"),
        pl.col("close").shift(1).over("ts_code").alias("prev_close"),
        pl.max_horizontal(pl.col("actual_atr"), pl.col("close") * 1e-4).replace(0.0, None).alias("atr_safe"),
    )
    df = df.with_columns(
        ((pl.col("actual_position") == 1) & ((pl.col("prev_position") == 0) | pl.col("prev_position").is_null())).alias("entry_day"),
        ((pl.col("actual_position") == 0) & (pl.col("prev_position") == 1)).alias("exit_day"),
    )

    # 4) 账户模拟：按日迭代（Equity 依赖前日状态，难以纯向量化）
    # 按 trade_date 分组、按日期顺序处理
    df = df.sort("trade_date")
    cash = float(cfg.initial_capital)
    hold_shares: dict[str, int] = {}
    buy_cost_basis: dict[str, float] = {}
    last_close: dict[str, float] = {}
    trade_pnls: list[float] = []
    daily_records: list[dict] = []
    needed = ["ts_code", "open", "close", "prev_close", "actual_position", "actual_atr", "atr_safe", "entry_day", "exit_day"]

    for (trade_date,), rows in df.group_by("trade_date", maintain_order=True):
        row_list = list(rows.select(needed).iter_rows(named=True))
        code_to_close = {r["ts_code"]: r["close"] for r in row_list}

        # 当日开盘前总资产：现金 + 持仓市值（用昨日收盘价）
        equity_start = cash
        for code, n in hold_shares.items():
            if n <= 0:
                continue
            equity_start += n * last_close.get(code, 0.0)

        # 先处理出场：卖出全部，现金增加，扣手续费；记录单笔盈亏
        for r in row_list:
            code = r["ts_code"]
            if not r["exit_day"] or hold_shares.get(code, 0) <= 0:
                continue
            n = hold_shares[code]
            price = r["open"]
            sell_amount = n * price
            cost = sell_amount * cfg.cost_roundtrip
            cash += sell_amount - cost
            cost_basis = buy_cost_basis.get(code, 0.0)
            trade_pnl = sell_amount - cost_basis - cost
            trade_pnls.append(trade_pnl)
            del hold_shares[code]
            if code in buy_cost_basis:
                del buy_cost_basis[code]

        # 建仓：Risk_Amount = 当日开盘前总资产 × risk_pct；目标股数 floor(Risk/ATR/100)*100，资金约束
        risk_amount = equity_start * cfg.risk_pct
        for r in row_list:
            code = r["ts_code"]
            if not r["entry_day"]:
                continue
            atr = r.get("atr_safe") or r["actual_atr"] or (r["close"] or 1.0) * 1e-4
            if atr is None or atr <= 0:
                continue
            # 【核心】风险金额转股数：Target_Shares = floor(Risk_Amount / actual_atr / 100) * 100
            target = int(risk_amount / float(atr) / LOT_SIZE) * LOT_SIZE
            if target < LOT_SIZE:
                continue
            price = r["open"]
            need_cash = target * price
            cost_buy = need_cash * cfg.cost_roundtrip
            if need_cash + cost_buy > cash:
                max_afford = int(cash / (price * (1 + cfg.cost_roundtrip))) // LOT_SIZE * LOT_SIZE
                target = min(target, max_afford)
                if target < LOT_SIZE:
                    continue
                need_cash = target * price
                cost_buy = need_cash * cfg.cost_roundtrip
            cash -= need_cash + cost_buy
            hold_shares[code] = target
            buy_cost_basis[code] = need_cash + cost_buy

        for code, c in code_to_close.items():
            last_close[code] = c

        # 每日总资产 = 现金 + 持仓市值（有当日收盘用当日，否则用昨收）
        equity_end = cash
        for code, n in hold_shares.items():
            if n <= 0:
                continue
            price_today = code_to_close.get(code, last_close.get(code, 0.0))
            equity_end += n * price_today
        daily_pnl = equity_end - equity_start
        strategy_return = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
        strategy_return = max(-0.99, min(10.0, strategy_return))
        daily_records.append({
            "trade_date": trade_date,
            "equity_start": equity_start,
            "equity_end": equity_end,
            "strategy_return": strategy_return,
        })

    # 存储单笔盈亏列表供 print_stats 使用（通过 config 或全局暂存）
    daily = pl.DataFrame(daily_records).sort("trade_date")
    daily = daily.with_columns(
        pl.col("strategy_return").fill_nan(0.0).clip(-0.99, 10.0)
    )
    daily = daily.with_columns((1.0 + pl.col("strategy_return")).cum_prod().alias("cum_nav"))
    return daily, trade_pnls


def print_stats(
    daily: pl.DataFrame,
    trade_pnls: list[float] | None = None,
    initial_capital: float = 10_000_000,
) -> None:
    """打印回测统计：总收益、最大回撤、胜率、Calmar、单笔最大亏损额。"""
    if daily.is_empty():
        print("无交易日数据。")
        return
    nav = daily["cum_nav"]
    if nav.null_count() > 0 or (nav.min() is not None and float(nav.min()) <= 0):
        nav = nav.fill_nan(1.0).clip(1e-8, 1e10)
    total_return = float(nav[-1]) - 1.0
    peak = nav.cum_max()
    max_dd = float(((nav / peak) - 1.0).min())
    win_rate = float((daily["strategy_return"] > 0).mean())

    # 年化收益（按 252 交易日）、Calmar = 年化收益 / |最大回撤|
    n_days = len(daily)
    if n_days >= 1:
        years = n_days / 252.0
        ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return
    else:
        ann_return = 0.0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # 单笔最大亏损额（平仓盈亏中的最小值）
    risk_unit = initial_capital * 0.01
    if trade_pnls:
        max_single_loss = min(trade_pnls)
        max_single_loss_pct = (max_single_loss / initial_capital * 100) if initial_capital else 0.0
    else:
        max_single_loss = None
        max_single_loss_pct = None

    print("=== 海龟回测结果（账户模拟法）===")
    print(f"总收益率:     {total_return:.4f}")
    print(f"年化收益:     {ann_return:.4f}")
    print(f"最大回撤:     {max_dd:.4f}")
    print(f"Calmar 比率:  {calmar:.4f}")
    print(f"胜率(日):     {win_rate:.4f}")
    if max_single_loss is not None:
        print(f"单笔最大亏损: {max_single_loss:,.0f} 元 ({max_single_loss_pct:.2f}%)")
        print(f"  (理论单笔风险约 {risk_unit:,.0f} 元，即 {initial_capital*0.01/initial_capital*100:.1f}%)")


def plot_nav(daily: pl.DataFrame, path: str = "turtle_nav.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return
    d = daily.to_pandas()
    d["trade_date"] = d["trade_date"].astype(str)
    if d["trade_date"].str.len().max() == 8:
        d["trade_date"] = (
            d["trade_date"].str[:4] + "-" + d["trade_date"].str[4:6] + "-" + d["trade_date"].str[6:8]
        )
    d["trade_date"] = d["trade_date"].astype("datetime64[ns]")
    plt.figure(figsize=(12, 5))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5)
    plt.title("Turtle Strategy NAV")
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
        risk_pct=0.01,
        cost_roundtrip=0.0003,
        t_plus_one=True,
    )

    # 优先从 shiming_daily_base.duckdb 读日线（前复权）；否则用合成数据
    db_path = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)

    if os.path.isfile(db_path):
        try:
            df = load_from_duckdb(db_path, config.start_date, config.end_date)
            if df.is_empty():
                raise ValueError("DuckDB 查询结果为空")
            print(f"已从 {db_path} 加载 {len(df):,} 条日线，{df['ts_code'].n_unique()} 只标的。")
        except Exception as e:
            print(f"从 DuckDB 加载失败（{e}），改用合成数据。")
            df = None
    else:
        print(f"未找到数据库 {db_path}，改用合成数据。")
        df = None

    if df is None:
        from datetime import timedelta
        n = 1200
        base = datetime(2015, 1, 1)
        close = [10.0 + 0.2 * i for i in range(n)]
        df = pl.DataFrame({
            "ts_code": ["000001.SZ"] * n,
            "trade_date": [(base + timedelta(days=k)).strftime("%Y%m%d") for k in range(n)],
            "open": [c - 0.05 for c in close],
            "high": [c + 0.08 for c in close],
            "low": [c - 0.08 for c in close],
            "close": close,
        })

    daily, trade_pnls = run_backtest(df, config)
    print_stats(daily, trade_pnls=trade_pnls, initial_capital=config.initial_capital)
    plot_nav(daily)
