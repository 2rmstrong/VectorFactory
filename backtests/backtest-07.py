"""
配对交易回测框架 · 月度动态挖掘 + 每日盯盘撮合（T+1）

模块：
1) 月末动态雷达：formation=250 天，Top500 流动性池 → corr>0.85 → coint(p<0.05) → Top100 pairs
2) 每日信号生成：对当月 Active Pairs 计算滚动 Beta/Spread/Z-Score（用于 T+1 撮合）
3) 日度撮合引擎：昨日信号，今日 open 成交；先平仓后开仓；50 个配对卡槽
4) 评价输出：净值曲线 + 年化/回撤/夏普 + 平均持仓天数 + 胜率
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)
import project_paths as pp

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


LOT_SIZE = 100


@dataclass(frozen=True)
class BacktestConfig:
    duckdb_path: str = "shiming_daily_base.duckdb"
    start_date: str = "20240101"
    end_date: str | None = None

    initial_capital: float = 10_000_000

    # 卡槽模型
    max_pair_positions: int = 50
    slot_capital: float = 200_000.0  # 满仓时约 20 万/对

    # 月度挖掘参数
    formation_window: int = 250
    top_n_liquid: int = 500
    corr_th: float = 0.85
    coint_p_th: float = 0.05
    top_k_pairs: int = 100
    liquidity_metric: str = "amount"  # amount | total_mv
    max_workers: int | None = None

    # 日度监控参数
    beta_window: int = 120
    z_window: int = 20
    entry_z: float = 2.0
    exit_band: float = 0.5
    stop_z: float = 4.0
    time_stop_days: int = 20

    # 交易摩擦（万三，双边都按同一个比例扣）
    fee: float = 0.0003

    # A 股融券受限：False 时仅保留框架（short leg 置 0）
    allow_shorting: bool = False


@dataclass
class PairPosition:
    stock_a: str
    stock_b: str
    direction: int  # 1: 多A空B；-1: 空A多B

    entry_date: str
    entry_open_a: float
    entry_open_b: float
    entry_z: float

    slot_capital: float
    shares_long: int
    shares_short: int
    long_leg: str
    short_leg: str

    holding_days: int = 0

    def key(self) -> tuple[str, str]:
        return (self.stock_a, self.stock_b) if self.stock_a <= self.stock_b else (self.stock_b, self.stock_a)


def _ensure_abs(path: str) -> str:
    return pp.resolve_db_path(path)


def _month_ends(trade_dates: list[str]) -> list[str]:
    if not trade_dates:
        return []
    td = pd.Series(trade_dates)
    m = td.str.slice(0, 6)
    idx = m.ne(m.shift(-1))
    return td[idx].tolist()


def _read_pl(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pl.DataFrame:
    try:
        if params:
            return con.execute(sql, params).pl()
        return con.execute(sql).pl()
    except Exception:
        if params:
            return pl.from_arrow(con.execute(sql, params).arrow())
        return pl.from_arrow(con.execute(sql).arrow())


def _pick_top_liquid(df: pl.DataFrame, top_n: int, metric: str) -> list[str]:
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

    ranked = (
        df.group_by("ts_code")
        .agg(pl.col(use_metric).mean().alias("_liq"))
        .sort("_liq", descending=True)
        .head(int(top_n))
        .select("ts_code")
    )
    return ranked.get_column("ts_code").to_list()


def _build_close_panel(df: pl.DataFrame, codes: list[str]) -> pd.DataFrame:
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
    wide = x.pivot(values="close", index="trade_date", on="ts_code", aggregate_function="first")
    pdf = wide.to_pandas()
    pdf.index = pd.to_datetime(pdf.index.astype(str), format="%Y%m%d", errors="coerce")
    pdf = pdf.sort_index()
    pdf = pdf.apply(pd.to_numeric, errors="coerce")
    pdf = pdf.dropna(axis=1, how="all")
    return pdf


def _corr_candidates(close_panel: pd.DataFrame, corr_th: float) -> list[tuple[str, str, float]]:
    logp = np.log(close_panel.where(close_panel > 0))
    corr = logp.corr(method="pearson", min_periods=max(60, int(len(logp) * 0.6)))
    cols = corr.columns.to_list()
    mat = corr.to_numpy()

    out: list[tuple[str, str, float]] = []
    n = len(cols)
    for i in range(n):
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
    mask = np.isfinite(s_a) & np.isfinite(s_b)
    s_a = s_a[mask]
    s_b = s_b[mask]
    if s_a.size < 160:
        return a, b, float("nan")
    try:
        _t, p, _crit = coint(s_a, s_b)
        return a, b, float(p)
    except Exception:
        return a, b, float("nan")


def _coint_parallel(
    close_panel: pd.DataFrame,
    pairs: list[tuple[str, str, float]],
    *,
    max_workers: int | None,
    desc: str,
) -> list[tuple[str, str, float]]:
    if not pairs:
        return []
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
            out.append(f.result())
    return out


def _compute_last_two_zscores(
    close_a: pd.Series,
    close_b: pd.Series,
    *,
    beta_window: int,
    z_window: int,
) -> tuple[float | None, float | None]:
    x = close_a.astype("float64")
    y = close_b.astype("float64")
    df = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if len(df) < max(beta_window, z_window) + 5:
        return None, None

    a = df["a"]
    b = df["b"]
    mean_a = a.rolling(beta_window).mean()
    mean_b = b.rolling(beta_window).mean()
    mean_ab = (a * b).rolling(beta_window).mean()
    mean_bb = (b * b).rolling(beta_window).mean()
    cov_ab = mean_ab - mean_a * mean_b
    var_b = mean_bb - mean_b * mean_b
    beta = cov_ab / var_b.replace(0.0, np.nan)

    spread = a - beta * b
    m = spread.rolling(z_window).mean()
    s = spread.rolling(z_window).std()
    z = (spread - m) / s.replace(0.0, np.nan)

    z = z.dropna()
    if len(z) < 2:
        return None, None
    return float(z.iloc[-1]), float(z.iloc[-2])


class DynamicPairsBacktester:
    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()

        self.df: pl.DataFrame | None = None
        self.trade_dates: list[str] = []
        self.month_ends: set[str] = set()

        self.active_pairs: list[tuple[str, str]] = []
        self.next_month_pairs: list[tuple[str, str]] | None = None

        self.cash: float = float(self.cfg.initial_capital)
        self.positions: dict[tuple[str, str], PairPosition] = {}
        self.closed_trades: list[dict] = []
        self.daily_records: list[dict] = []

        self._open_px: dict[str, float] = {}
        self._close_px: dict[str, float] = {}

    def load_from_duckdb(self) -> pl.DataFrame:
        end = self.cfg.end_date or datetime.now().strftime("%Y%m%d")
        db_path = _ensure_abs(self.cfg.duckdb_path)
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"未找到数据库：{db_path}")

        con = duckdb.connect(db_path, read_only=True)
        try:
            d = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, open, close, amount
                FROM daily_data
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
            b = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, total_mv
                FROM daily_basic
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
            a = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, adj_factor
                FROM adj_factor
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
        finally:
            con.close()

        j = (
            d.join(a, on=["ts_code", "trade_date"], how="inner")
            .join(b, on=["ts_code", "trade_date"], how="left")
            .sort(["ts_code", "trade_date"])
        )

        last_af = pl.col("adj_factor").last().over("ts_code")
        ratio = pl.col("adj_factor") / last_af

        out = j.with_columns(
            [
                (pl.col("open") * ratio).alias("open"),
                (pl.col("close") * ratio).alias("close"),
                pl.col("amount").cast(pl.Float64).fill_nan(None),
                pl.col("total_mv").cast(pl.Float64).fill_nan(None),
            ]
        ).select(["ts_code", "trade_date", "open", "close", "amount", "total_mv"])

        out = out.drop_nulls(["ts_code", "trade_date", "open", "close"]).sort(["trade_date", "ts_code"])
        self.df = out
        self.trade_dates = out.get_column("trade_date").unique().sort().to_list()
        self.month_ends = set(_month_ends(self.trade_dates))
        return out

    def _monthly_rebalance(self, month_end_date: str) -> list[tuple[str, str]]:
        assert self.df is not None
        td_pos = {d: i for i, d in enumerate(self.trade_dates)}
        i = td_pos.get(month_end_date)
        if i is None or i + 1 < self.cfg.formation_window:
            return []
        window_dates = self.trade_dates[i + 1 - self.cfg.formation_window : i + 1]
        w = self.df.filter(pl.col("trade_date").is_in(window_dates))

        top_codes = _pick_top_liquid(w, self.cfg.top_n_liquid, self.cfg.liquidity_metric)
        w = w.filter(pl.col("ts_code").is_in(top_codes))

        panel = _build_close_panel(w, top_codes)
        min_obs = int(self.cfg.formation_window * 0.8)
        panel = panel.loc[:, panel.notna().sum(axis=0) >= min_obs]
        if panel.shape[1] < 2:
            return []

        candidates = _corr_candidates(panel, self.cfg.corr_th)
        if not candidates:
            return []

        coint_res = _coint_parallel(
            panel,
            candidates,
            max_workers=self.cfg.max_workers,
            desc=f"ADF(coint) {month_end_date}",
        )
        valid = [(a, b, p) for a, b, p in coint_res if np.isfinite(p) and p < self.cfg.coint_p_th]
        if not valid:
            return []

        valid.sort(key=lambda x: x[2])
        topk = valid[: int(self.cfg.top_k_pairs)]
        pairs = []
        for a, b, _p in topk:
            aa, bb = (a, b) if a <= b else (b, a)
            pairs.append((aa, bb))
        return pairs

    def _get_close_series(self, codes: Iterable[str], end_date: str, lookback_days: int) -> dict[str, pd.Series]:
        assert self.df is not None
        td_pos = {d: i for i, d in enumerate(self.trade_dates)}
        i = td_pos.get(end_date)
        if i is None:
            return {}
        window_dates = self.trade_dates[max(0, i + 1 - lookback_days) : i + 1]
        sub = self.df.filter(pl.col("trade_date").is_in(window_dates) & pl.col("ts_code").is_in(list(codes)))
        if sub.is_empty():
            return {}
        pdf = sub.select(["trade_date", "ts_code", "close"]).to_pandas()
        pdf["trade_date"] = pd.to_datetime(pdf["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        out = {}
        for code, g in pdf.groupby("ts_code"):
            s = g.sort_values("trade_date").set_index("trade_date")["close"].astype("float64")
            out[str(code)] = s
        return out

    def _compute_pair_signal(self, stock_a: str, stock_b: str, prev_date: str) -> dict:
        lookback = max(self.cfg.formation_window, self.cfg.beta_window + self.cfg.z_window + 10, 200)
        series = self._get_close_series([stock_a, stock_b], prev_date, lookback_days=lookback)
        s_a = series.get(stock_a)
        s_b = series.get(stock_b)
        if s_a is None or s_b is None:
            return {"z": None, "z_prev": None, "entry_dir": 0, "exit_flag": False, "stop_flag": False}

        z_t, z_tm1 = _compute_last_two_zscores(s_a, s_b, beta_window=self.cfg.beta_window, z_window=self.cfg.z_window)
        if z_t is None or z_tm1 is None:
            return {"z": None, "z_prev": None, "entry_dir": 0, "exit_flag": False, "stop_flag": False}

        entry_dir = 0
        if z_t > self.cfg.entry_z and z_tm1 <= self.cfg.entry_z:
            entry_dir = -1
        elif z_t < -self.cfg.entry_z and z_tm1 >= -self.cfg.entry_z:
            entry_dir = 1

        exit_flag = (-self.cfg.exit_band <= z_t <= self.cfg.exit_band)
        stop_flag = (z_t > self.cfg.stop_z) or (z_t < -self.cfg.stop_z)
        return {"z": float(z_t), "z_prev": float(z_tm1), "entry_dir": entry_dir, "exit_flag": exit_flag, "stop_flag": stop_flag}

    def _open_position(self, a: str, b: str, direction: int, trade_date: str, z: float) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key in self.positions:
            return False
        if len(self.positions) >= int(self.cfg.max_pair_positions):
            return False
        if self.cash < float(self.cfg.slot_capital):
            return False

        open_a = float(self._open_px.get(a, 0.0))
        open_b = float(self._open_px.get(b, 0.0))
        if open_a <= 0 or open_b <= 0:
            return False

        leg_cap = float(self.cfg.slot_capital) * 0.5
        shares_a = int(math.floor(leg_cap / (open_a * LOT_SIZE)) * LOT_SIZE)
        shares_b = int(math.floor(leg_cap / (open_b * LOT_SIZE)) * LOT_SIZE)
        if shares_a <= 0 or shares_b <= 0:
            return False

        if direction == 1:
            long_leg, short_leg = a, b
            shares_long = shares_a
            shares_short = shares_b if self.cfg.allow_shorting else 0
        else:
            long_leg, short_leg = b, a
            shares_long = shares_b
            shares_short = shares_a if self.cfg.allow_shorting else 0

        entry_notional = shares_long * float(self._open_px[long_leg]) + shares_short * float(self._open_px[short_leg])
        fee = entry_notional * float(self.cfg.fee)
        if self.cash < float(self.cfg.slot_capital) + fee:
            return False

        self.cash -= float(self.cfg.slot_capital) + fee
        pos = PairPosition(
            stock_a=a,
            stock_b=b,
            direction=int(direction),
            entry_date=trade_date,
            entry_open_a=open_a,
            entry_open_b=open_b,
            entry_z=float(z),
            slot_capital=float(self.cfg.slot_capital),
            shares_long=int(shares_long),
            shares_short=int(shares_short),
            long_leg=long_leg,
            short_leg=short_leg,
            holding_days=0,
        )
        self.positions[key] = pos
        return True

    def _close_position(self, key: tuple[str, str], trade_date: str, reason: str, z: float | None) -> None:
        pos = self.positions.get(key)
        if pos is None:
            return

        open_a = float(self._open_px.get(pos.stock_a, 0.0))
        open_b = float(self._open_px.get(pos.stock_b, 0.0))
        if open_a <= 0 or open_b <= 0:
            return

        exit_px = {pos.stock_a: open_a, pos.stock_b: open_b}
        entry_px = {pos.stock_a: float(pos.entry_open_a), pos.stock_b: float(pos.entry_open_b)}

        long_exit = float(exit_px[pos.long_leg])
        short_exit = float(exit_px[pos.short_leg])
        long_entry = float(entry_px[pos.long_leg])
        short_entry = float(entry_px[pos.short_leg])

        pnl_long = float(pos.shares_long) * (long_exit - long_entry)
        pnl_short = float(pos.shares_short) * (short_entry - short_exit)

        exit_notional = float(pos.shares_long) * long_exit + float(pos.shares_short) * short_exit
        fee = exit_notional * float(self.cfg.fee)
        pnl = pnl_long + pnl_short - fee

        self.cash += float(pos.slot_capital) + pnl
        self.closed_trades.append(
            {
                "pair": f"{pos.stock_a}|{pos.stock_b}",
                "stock_A": pos.stock_a,
                "stock_B": pos.stock_b,
                "direction": pos.direction,
                "entry_date": pos.entry_date,
                "exit_date": trade_date,
                "holding_days": int(pos.holding_days),
                "entry_z": float(pos.entry_z),
                "exit_z": (float(z) if z is not None else None),
                "pnl": float(pnl),
                "reason": reason,
            }
        )
        self.positions.pop(key, None)

    def _mark_to_market(self) -> float:
        total = float(self.cash)
        for pos in self.positions.values():
            close_a = float(self._close_px.get(pos.stock_a, pos.entry_open_a))
            close_b = float(self._close_px.get(pos.stock_b, pos.entry_open_b))
            close_px = {pos.stock_a: close_a, pos.stock_b: close_b}
            entry_px = {pos.stock_a: float(pos.entry_open_a), pos.stock_b: float(pos.entry_open_b)}

            long_close = float(close_px[pos.long_leg])
            short_close = float(close_px[pos.short_leg])
            long_entry = float(entry_px[pos.long_leg])
            short_entry = float(entry_px[pos.short_leg])

            upnl_long = float(pos.shares_long) * (long_close - long_entry)
            upnl_short = float(pos.shares_short) * (short_entry - short_close)
            total += float(pos.slot_capital) + (upnl_long + upnl_short)
        return total

    def run(self) -> pl.DataFrame:
        if self.df is None:
            self.load_from_duckdb()
        assert self.df is not None

        self.next_month_pairs = None
        self.active_pairs = []

        df_iter = self.df.select(["trade_date", "ts_code", "open", "close"]).sort(["trade_date", "ts_code"])
        unique_dates = df_iter.get_column("trade_date").unique().sort().to_list()

        for di, trade_date in enumerate(tqdm(unique_dates, desc="Backtest days")):
            day = df_iter.filter(pl.col("trade_date") == trade_date)
            rows = list(day.select(["ts_code", "open", "close"]).iter_rows(named=True))
            self._open_px = {r["ts_code"]: float(r["open"]) for r in rows if r["open"] is not None}
            self._close_px = {r["ts_code"]: float(r["close"]) for r in rows if r["close"] is not None}

            if di == 0:
                equity = self._mark_to_market()
                self.daily_records.append({"trade_date": trade_date, "total_equity": equity, "strategy_return": 0.0, "pairs_held": len(self.positions)})
                if trade_date in self.month_ends:
                    self.next_month_pairs = self._monthly_rebalance(trade_date)
                continue

            prev_date = unique_dates[di - 1]
            if trade_date[:6] != prev_date[:6]:
                self.active_pairs = self.next_month_pairs or []

            equity_start = self._mark_to_market()

            # Exit
            exit_actions: list[tuple[tuple[str, str], str]] = []
            for key, pos in list(self.positions.items()):
                if self.active_pairs and (key not in self.active_pairs):
                    exit_actions.append((key, "drop_from_top100"))
                    continue

                sig = self._compute_pair_signal(pos.stock_a, pos.stock_b, prev_date)
                z = sig["z"]
                if z is None:
                    continue
                if bool(sig["exit_flag"]):
                    exit_actions.append((key, "mean_revert"))
                elif bool(sig["stop_flag"]):
                    exit_actions.append((key, "z_stop"))
                elif int(pos.holding_days) >= int(self.cfg.time_stop_days):
                    exit_actions.append((key, "time_stop"))

            for key, reason in exit_actions:
                a, b = key
                z_now = self._compute_pair_signal(a, b, prev_date).get("z")
                self._close_position(key, trade_date, reason, z_now)

            # Entry
            free_slots = int(self.cfg.max_pair_positions) - len(self.positions)
            if free_slots > 0 and self.active_pairs:
                cands = []
                for a, b in self.active_pairs:
                    key = (a, b) if a <= b else (b, a)
                    if key in self.positions:
                        continue
                    sig = self._compute_pair_signal(a, b, prev_date)
                    z = sig["z"]
                    d = int(sig["entry_dir"])
                    if z is None or d == 0:
                        continue
                    cands.append((a, b, d, float(z)))
                cands.sort(key=lambda x: abs(x[3]), reverse=True)
                cands = cands[:free_slots]
                for a, b, d, z in cands:
                    self._open_position(a, b, d, trade_date, z)

            for pos in self.positions.values():
                pos.holding_days += 1

            equity_end = self._mark_to_market()
            daily_pnl = equity_end - equity_start
            daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
            daily_ret = max(-0.99, min(10.0, float(daily_ret)))
            self.daily_records.append(
                {"trade_date": trade_date, "total_equity": equity_end, "strategy_return": daily_ret, "pairs_held": len(self.positions)}
            )

            if trade_date in self.month_ends:
                self.next_month_pairs = self._monthly_rebalance(trade_date)

        daily = pl.DataFrame(self.daily_records).sort("trade_date")
        daily = daily.with_columns((pl.col("total_equity") / pl.col("total_equity").first()).alias("cum_nav"))
        return daily

    def stats(self, daily: pl.DataFrame) -> dict[str, float]:
        if daily.is_empty():
            return {"total_return": 0.0, "ann_return": 0.0, "max_dd": 0.0, "sharpe": 0.0, "avg_hold_days": 0.0, "win_rate": 0.0}

        nav = daily["cum_nav"].fill_nan(1.0).clip(1e-8, 1e12)
        total_return = float(nav[-1]) - 1.0
        peak = nav.cum_max()
        max_dd = float(((nav / peak) - 1.0).min())

        n_days = len(daily)
        years = n_days / 252.0 if n_days else 0.0
        ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return

        ret = daily["strategy_return"].fill_nan(0.0)
        mu = float(ret.mean())
        sigma = float(ret.std()) if ret.len() > 1 else 0.0
        sharpe = (mu / sigma * math.sqrt(252.0)) if sigma > 1e-12 else 0.0

        if self.closed_trades:
            hold_days = [int(t["holding_days"]) for t in self.closed_trades if t.get("holding_days") is not None]
            avg_hold_days = float(np.mean(hold_days)) if hold_days else 0.0
            pnls = [float(t["pnl"]) for t in self.closed_trades]
            win_rate = float(np.mean([1.0 if p > 0 else 0.0 for p in pnls])) if pnls else 0.0
        else:
            avg_hold_days = 0.0
            win_rate = 0.0

        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "avg_hold_days": avg_hold_days,
            "win_rate": win_rate,
        }


def plot_nav(daily: pl.DataFrame, path: str = "pairs_nav.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    if daily.is_empty():
        print("无净值数据，跳过绘图。")
        return

    d = daily.select(["trade_date", "cum_nav"]).to_pandas()
    d["trade_date"] = d["trade_date"].astype(str)
    if d["trade_date"].str.len().max() == 8:
        d["trade_date"] = d["trade_date"].str[:4] + "-" + d["trade_date"].str[4:6] + "-" + d["trade_date"].str[6:8]
    d["trade_date"] = d["trade_date"].astype("datetime64[ns]")

    plt.figure(figsize=(12, 5))
    plt.plot(d["trade_date"], d["cum_nav"], lw=1.5)
    plt.title("Dynamic Pairs Trading NAV (Monthly Mining + Daily Monitor, T+1)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存: {path}")


def _ensure_abs(path: str) -> str:
    return pp.resolve_db_path(path)


def _month_ends(trade_dates: list[str]) -> list[str]:
    if not trade_dates:
        return []
    td = pd.Series(trade_dates)
    m = td.str.slice(0, 6)
    idx = m.ne(m.shift(-1))
    return td[idx].tolist()


def _read_pl(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pl.DataFrame:
    try:
        if params:
            return con.execute(sql, params).pl()
        return con.execute(sql).pl()
    except Exception:
        if params:
            return pl.from_arrow(con.execute(sql, params).arrow())
        return pl.from_arrow(con.execute(sql).arrow())


def _pick_top_liquid(df: pl.DataFrame, top_n: int, metric: str) -> list[str]:
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

    ranked = (
        df.group_by("ts_code")
        .agg(pl.col(use_metric).mean().alias("_liq"))
        .sort("_liq", descending=True)
        .head(int(top_n))
        .select("ts_code")
    )
    return ranked.get_column("ts_code").to_list()


def _build_close_panel(df: pl.DataFrame, codes: list[str]) -> pd.DataFrame:
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
    wide = x.pivot(values="close", index="trade_date", on="ts_code", aggregate_function="first")
    pdf = wide.to_pandas()
    pdf.index = pd.to_datetime(pdf.index.astype(str), format="%Y%m%d", errors="coerce")
    pdf = pdf.sort_index()
    pdf = pdf.apply(pd.to_numeric, errors="coerce")
    pdf = pdf.dropna(axis=1, how="all")
    return pdf


def _corr_candidates(close_panel: pd.DataFrame, corr_th: float) -> list[tuple[str, str, float]]:
    logp = np.log(close_panel.where(close_panel > 0))
    corr = logp.corr(method="pearson", min_periods=max(60, int(len(logp) * 0.6)))
    cols = corr.columns.to_list()
    mat = corr.to_numpy()

    out: list[tuple[str, str, float]] = []
    n = len(cols)
    for i in range(n):
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
    mask = np.isfinite(s_a) & np.isfinite(s_b)
    s_a = s_a[mask]
    s_b = s_b[mask]
    if s_a.size < 160:
        return a, b, float("nan")
    try:
        _t, p, _crit = coint(s_a, s_b)
        return a, b, float(p)
    except Exception:
        return a, b, float("nan")


def _coint_parallel(
    close_panel: pd.DataFrame,
    pairs: list[tuple[str, str, float]],
    *,
    max_workers: int | None,
    desc: str,
) -> list[tuple[str, str, float]]:
    if not pairs:
        return []
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
            out.append(f.result())
    return out


def _compute_last_two_zscores(
    close_a: pd.Series,
    close_b: pd.Series,
    *,
    beta_window: int,
    z_window: int,
) -> tuple[float | None, float | None]:
    """
    返回 (z_t, z_t-1)；若不足样本则返回 None。
    """
    x = close_a.astype("float64")
    y = close_b.astype("float64")
    df = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if len(df) < max(beta_window, z_window) + 5:
        return None, None

    a = df["a"]
    b = df["b"]
    mean_a = a.rolling(beta_window).mean()
    mean_b = b.rolling(beta_window).mean()
    mean_ab = (a * b).rolling(beta_window).mean()
    mean_bb = (b * b).rolling(beta_window).mean()
    cov_ab = mean_ab - mean_a * mean_b
    var_b = mean_bb - mean_b * mean_b
    beta = cov_ab / var_b.replace(0.0, np.nan)

    spread = a - beta * b
    m = spread.rolling(z_window).mean()
    s = spread.rolling(z_window).std()
    z = (spread - m) / s.replace(0.0, np.nan)

    z = z.dropna()
    if len(z) < 2:
        return None, None
    return float(z.iloc[-1]), float(z.iloc[-2])


class DynamicPairsBacktester:
    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()

        self.df: pl.DataFrame | None = None
        self.trade_dates: list[str] = []
        self.month_ends: set[str] = set()

        self.active_pairs: list[tuple[str, str]] = []
        self.next_month_pairs: list[tuple[str, str]] | None = None

        self.cash: float = float(self.cfg.initial_capital)
        self.positions: dict[tuple[str, str], PairPosition] = {}
        self.closed_trades: list[dict] = []

        self.daily_records: list[dict] = []

        # 当日行情缓存（长表按 trade_date 分组迭代时生成）
        self._open_px: dict[str, float] = {}
        self._close_px: dict[str, float] = {}
        # 昨日收盘缓存：用于“开盘前估值”（避免用到当日收盘，导致日收益口径错位）
        self._last_close: dict[str, float] = {}

    # -------------------------
    # Data loading
    # -------------------------
    def load_from_duckdb(self) -> pl.DataFrame:
        end = self.cfg.end_date or datetime.now().strftime("%Y%m%d")
        db_path = _ensure_abs(self.cfg.duckdb_path)
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"未找到数据库：{db_path}")

        con = duckdb.connect(db_path, read_only=True)
        try:
            d = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, open, close, amount
                FROM daily_data
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
            b = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, total_mv
                FROM daily_basic
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
            a = _read_pl(
                con,
                """
                SELECT ts_code, trade_date, adj_factor
                FROM adj_factor
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                [self.cfg.start_date, end],
            )
        finally:
            con.close()

        j = (
            d.join(a, on=["ts_code", "trade_date"], how="inner")
            .join(b, on=["ts_code", "trade_date"], how="left")
            .sort(["ts_code", "trade_date"])
        )

        last_af = pl.col("adj_factor").last().over("ts_code")
        ratio = pl.col("adj_factor") / last_af

        out = j.with_columns(
            [
                (pl.col("open") * ratio).alias("open"),
                (pl.col("close") * ratio).alias("close"),
                pl.col("amount").cast(pl.Float64).fill_nan(None),
                pl.col("total_mv").cast(pl.Float64).fill_nan(None),
            ]
        ).select(["ts_code", "trade_date", "open", "close", "amount", "total_mv"])

        out = out.drop_nulls(["ts_code", "trade_date", "open", "close"]).sort(["trade_date", "ts_code"])
        self.df = out
        self.trade_dates = out.get_column("trade_date").unique().sort().to_list()
        self.month_ends = set(_month_ends(self.trade_dates))
        return out

    # -------------------------
    # Monthly rebalance
    # -------------------------
    def _monthly_rebalance(self, month_end_date: str) -> list[tuple[str, str]]:
        assert self.df is not None
        td_pos = {d: i for i, d in enumerate(self.trade_dates)}
        i = td_pos.get(month_end_date)
        if i is None or i + 1 < self.cfg.formation_window:
            return []
        window_dates = self.trade_dates[i + 1 - self.cfg.formation_window : i + 1]
        w = self.df.filter(pl.col("trade_date").is_in(window_dates))

        # TopN 流动性池
        top_codes = _pick_top_liquid(w, self.cfg.top_n_liquid, self.cfg.liquidity_metric)
        w = w.filter(pl.col("ts_code").is_in(top_codes))

        panel = _build_close_panel(w, top_codes)
        min_obs = int(self.cfg.formation_window * 0.8)
        panel = panel.loc[:, panel.notna().sum(axis=0) >= min_obs]
        if panel.shape[1] < 2:
            return []

        candidates = _corr_candidates(panel, self.cfg.corr_th)
        if not candidates:
            return []

        coint_res = _coint_parallel(
            panel,
            candidates,
            max_workers=self.cfg.max_workers,
            desc=f"ADF(coint) {month_end_date}",
        )
        valid = [(a, b, p) for a, b, p in coint_res if np.isfinite(p) and p < self.cfg.coint_p_th]
        if not valid:
            return []

        valid.sort(key=lambda x: x[2])
        topk = valid[: int(self.cfg.top_k_pairs)]
        pairs = []
        for a, b, _p in topk:
            aa, bb = (a, b) if a <= b else (b, a)
            pairs.append((aa, bb))
        return pairs

    # -------------------------
    # Daily monitor (z-score)
    # -------------------------
    def _get_close_series(self, codes: Iterable[str], end_date: str, lookback_days: int) -> dict[str, pd.Series]:
        """
        为一组代码构建 close 的时间序列（按 trade_date 对齐），只取 end_date 往前 lookback_days 个交易日。
        """
        assert self.df is not None
        td_pos = {d: i for i, d in enumerate(self.trade_dates)}
        i = td_pos.get(end_date)
        if i is None:
            return {}
        window_dates = self.trade_dates[max(0, i + 1 - lookback_days) : i + 1]
        sub = self.df.filter(pl.col("trade_date").is_in(window_dates) & pl.col("ts_code").is_in(list(codes)))
        if sub.is_empty():
            return {}
        pdf = sub.select(["trade_date", "ts_code", "close"]).to_pandas()
        pdf["trade_date"] = pd.to_datetime(pdf["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        out = {}
        for code, g in pdf.groupby("ts_code"):
            s = g.sort_values("trade_date").set_index("trade_date")["close"].astype("float64")
            out[str(code)] = s
        return out

    def _compute_pair_signal(self, stock_a: str, stock_b: str, prev_date: str) -> dict:
        """
        用 prev_date（昨日收盘）生成信号（用于今日 open 撮合）。
        返回：{z, z_prev, entry_dir, exit_flag, stop_flag}
        """
        lookback = max(self.cfg.formation_window, self.cfg.beta_window + self.cfg.z_window + 10, 200)
        series = self._get_close_series([stock_a, stock_b], prev_date, lookback_days=lookback)
        s_a = series.get(stock_a)
        s_b = series.get(stock_b)
        if s_a is None or s_b is None:
            return {"z": None, "z_prev": None, "entry_dir": 0, "exit_flag": False, "stop_flag": False}

        z_t, z_tm1 = _compute_last_two_zscores(s_a, s_b, beta_window=self.cfg.beta_window, z_window=self.cfg.z_window)
        if z_t is None or z_tm1 is None:
            return {"z": None, "z_prev": None, "entry_dir": 0, "exit_flag": False, "stop_flag": False}

        entry_dir = 0
        if z_t > self.cfg.entry_z and z_tm1 <= self.cfg.entry_z:
            entry_dir = -1  # 空A多B
        elif z_t < -self.cfg.entry_z and z_tm1 >= -self.cfg.entry_z:
            entry_dir = 1   # 多A空B

        exit_flag = (-self.cfg.exit_band <= z_t <= self.cfg.exit_band)
        stop_flag = (z_t > self.cfg.stop_z) or (z_t < -self.cfg.stop_z)
        return {"z": float(z_t), "z_prev": float(z_tm1), "entry_dir": entry_dir, "exit_flag": exit_flag, "stop_flag": stop_flag}

    # -------------------------
    # Execution engine (T+1)
    # -------------------------
    def _open_position(self, a: str, b: str, direction: int, trade_date: str, z: float) -> bool:
        """
        以今日 open 开仓；占用一个卡槽（slot_capital）。
        """
        key = (a, b) if a <= b else (b, a)
        if key in self.positions:
            return False
        if len(self.positions) >= int(self.cfg.max_pair_positions):
            return False
        if self.cash < float(self.cfg.slot_capital):
            return False

        open_a = float(self._open_px.get(a, 0.0))
        open_b = float(self._open_px.get(b, 0.0))
        if open_a <= 0 or open_b <= 0:
            return False

        # 双边：每腿 50% notional；A 股无法做空时保留占位（shares_short=0）
        leg_cap = float(self.cfg.slot_capital) * 0.5
        shares_a = int(math.floor(leg_cap / (open_a * LOT_SIZE)) * LOT_SIZE)
        shares_b = int(math.floor(leg_cap / (open_b * LOT_SIZE)) * LOT_SIZE)
        if shares_a <= 0 or shares_b <= 0:
            return False

        if direction == 1:  # 多A空B
            long_leg, short_leg = a, b
            shares_long = shares_a
            shares_short = shares_b if self.cfg.allow_shorting else 0
        else:               # 空A多B
            long_leg, short_leg = b, a
            shares_long = shares_b
            shares_short = shares_a if self.cfg.allow_shorting else 0

        # 费用：按入场时成交额（双边）扣万三
        entry_notional = shares_long * float(self._open_px[long_leg]) + shares_short * float(self._open_px[short_leg])
        fee = entry_notional * float(self.cfg.fee)

        # 占用卡槽资金（slot_capital）+ 扣手续费
        if self.cash < float(self.cfg.slot_capital) + fee:
            return False

        self.cash -= float(self.cfg.slot_capital) + fee
        pos = PairPosition(
            stock_a=a,
            stock_b=b,
            direction=int(direction),
            entry_date=trade_date,
            entry_open_a=open_a,
            entry_open_b=open_b,
            entry_z=float(z),
            slot_capital=float(self.cfg.slot_capital),
            shares_long=int(shares_long),
            shares_short=int(shares_short),
            long_leg=long_leg,
            short_leg=short_leg,
            holding_days=0,
        )
        self.positions[key] = pos
        return True

    def _close_position(self, key: tuple[str, str], trade_date: str, reason: str, z: float | None) -> None:
        pos = self.positions.get(key)
        if pos is None:
            return

        # 以今日 open 平仓
        open_a = float(self._open_px.get(pos.stock_a, 0.0))
        open_b = float(self._open_px.get(pos.stock_b, 0.0))
        if open_a <= 0 or open_b <= 0:
            return

        exit_px = {pos.stock_a: open_a, pos.stock_b: open_b}
        entry_px = {pos.stock_a: float(pos.entry_open_a), pos.stock_b: float(pos.entry_open_b)}

        long_exit = float(exit_px[pos.long_leg])
        short_exit = float(exit_px[pos.short_leg])
        long_entry = float(entry_px[pos.long_leg])
        short_entry = float(entry_px[pos.short_leg])

        pnl_long = float(pos.shares_long) * (long_exit - long_entry)
        pnl_short = float(pos.shares_short) * (short_entry - short_exit)

        exit_notional = float(pos.shares_long) * long_exit + float(pos.shares_short) * short_exit
        fee = exit_notional * float(self.cfg.fee)
        pnl = pnl_long + pnl_short - fee

        self.cash += float(pos.slot_capital) + pnl

        self.closed_trades.append(
            {
                "pair": f"{pos.stock_a}|{pos.stock_b}",
                "stock_A": pos.stock_a,
                "stock_B": pos.stock_b,
                "direction": pos.direction,
                "entry_date": pos.entry_date,
                "exit_date": trade_date,
                "holding_days": int(pos.holding_days),
                "entry_z": float(pos.entry_z),
                "exit_z": (float(z) if z is not None else None),
                "pnl": float(pnl),
                "reason": reason,
            }
        )

        self.positions.pop(key, None)

    def _mark_to_market(self, price_map: dict[str, float] | None = None) -> float:
        """按给定 price_map（通常为昨收或今收）估值总资产。"""
        px = price_map if price_map is not None else self._close_px
        total = float(self.cash)
        for pos in self.positions.values():
            close_a = float(px.get(pos.stock_a, pos.entry_open_a))
            close_b = float(px.get(pos.stock_b, pos.entry_open_b))
            close_px = {pos.stock_a: close_a, pos.stock_b: close_b}
            entry_px = {pos.stock_a: float(pos.entry_open_a), pos.stock_b: float(pos.entry_open_b)}

            long_close = float(close_px[pos.long_leg])
            short_close = float(close_px[pos.short_leg])
            long_entry = float(entry_px[pos.long_leg])
            short_entry = float(entry_px[pos.short_leg])

            upnl_long = float(pos.shares_long) * (long_close - long_entry)
            upnl_short = float(pos.shares_short) * (short_entry - short_close)
            total += float(pos.slot_capital) + (upnl_long + upnl_short)
        return total

    # -------------------------
    # Main loop
    # -------------------------
    def run(self) -> pl.DataFrame:
        if self.df is None:
            self.load_from_duckdb()
        assert self.df is not None

        # 预先计算：月末当天收盘后决定次月 active_pairs
        self.next_month_pairs = None
        self.active_pairs = []

        # 分组迭代（按 trade_date）
        needed = ["ts_code", "open", "close"]
        df_iter = self.df.select(needed + ["trade_date"]).sort(["trade_date", "ts_code"])

        # 建立按 trade_date 的列表（确保稳定顺序）
        unique_dates = df_iter.get_column("trade_date").unique().sort().to_list()
        td_pos = {d: i for i, d in enumerate(unique_dates)}

        # 外层进度条：日度循环
        for di, trade_date in enumerate(tqdm(unique_dates, desc="Backtest days")):
            day = df_iter.filter(pl.col("trade_date") == trade_date)
            rows = list(day.select(["ts_code", "open", "close"]).iter_rows(named=True))
            self._open_px = {r["ts_code"]: float(r["open"]) for r in rows if r["open"] is not None}
            self._close_px = {r["ts_code"]: float(r["close"]) for r in rows if r["close"] is not None}

            # 首日无法 T+1 执行
            if di == 0:
                equity = self._mark_to_market(self._close_px)
                self.daily_records.append(
                    {"trade_date": trade_date, "total_equity": equity, "strategy_return": 0.0, "pairs_held": len(self.positions)}
                )
                # 建立昨收缓存
                self._last_close = dict(self._close_px)
                # 月末收盘后：生成次月 active pairs
                if trade_date in self.month_ends:
                    self.next_month_pairs = self._monthly_rebalance(trade_date)
                continue

            prev_date = unique_dates[di - 1]

            # 月切换：把上月末生成的 pairs 作为“本月监控名单”
            if trade_date[:6] != prev_date[:6]:
                self.active_pairs = self.next_month_pairs or []

            # 开盘前估值：严格用“昨收”估值（避免用到当日收盘产生口径错位）
            equity_start = self._mark_to_market(self._last_close)

            # --------------------------
            # Step 1：平仓 / 止损 / 强制踢出
            # --------------------------
            exit_keys: list[tuple[str, str]] = []
            for key, pos in list(self.positions.items()):
                # 换月强制踢出：不在本月 active_pairs
                if self.active_pairs and (key not in self.active_pairs):
                    exit_keys.append((key, "drop_from_top100"))
                    continue

                sig = self._compute_pair_signal(pos.stock_a, pos.stock_b, prev_date)
                z = sig["z"]
                if z is None:
                    continue

                # 1) 止盈回归
                if bool(sig["exit_flag"]):
                    exit_keys.append((key, "mean_revert"))
                    continue
                # 2) 空间止损
                if bool(sig["stop_flag"]):
                    exit_keys.append((key, "z_stop"))
                    continue
                # 3) 时间止损
                if int(pos.holding_days) >= int(self.cfg.time_stop_days):
                    exit_keys.append((key, "time_stop"))
                    continue

            # 先平仓释放卡槽
            for key, reason in exit_keys:
                a, b = key
                z_now = self._compute_pair_signal(a, b, prev_date).get("z")
                self._close_position(key, trade_date, reason, z_now)

            # --------------------------
            # Step 2：开仓进场（按 |z| 降序排队）
            # --------------------------
            free_slots = int(self.cfg.max_pair_positions) - len(self.positions)
            if free_slots > 0 and self.active_pairs:
                candidates = []
                for a, b in self.active_pairs:
                    key = (a, b) if a <= b else (b, a)
                    if key in self.positions:
                        continue
                    sig = self._compute_pair_signal(a, b, prev_date)
                    z = sig["z"]
                    d = int(sig["entry_dir"])
                    if z is None or d == 0:
                        continue
                    candidates.append((a, b, d, float(z)))

                candidates.sort(key=lambda x: abs(x[3]), reverse=True)
                if len(candidates) > free_slots:
                    candidates = candidates[:free_slots]

                for a, b, d, z in candidates:
                    self._open_position(a, b, d, trade_date, z)

            # 更新持仓天数（持有跨过一个交易日）
            for pos in self.positions.values():
                pos.holding_days += 1

            # 收盘估值：用当日收盘
            equity_end = self._mark_to_market(self._close_px)
            daily_pnl = equity_end - equity_start
            daily_ret = (daily_pnl / equity_start) if equity_start > 1e-8 else 0.0
            daily_ret = max(-0.99, min(10.0, float(daily_ret)))
            self.daily_records.append(
                {
                    "trade_date": trade_date,
                    "total_equity": equity_end,
                    "strategy_return": daily_ret,
                    "pairs_held": len(self.positions),
                }
            )

            # 月末收盘后：生成次月 active pairs（耗时，放最后）
            if trade_date in self.month_ends:
                self.next_month_pairs = self._monthly_rebalance(trade_date)

            # 更新昨收缓存（用于下一交易日开盘前估值）
            self._last_close = dict(self._close_px)

        daily = pl.DataFrame(self.daily_records).sort("trade_date")
        daily = daily.with_columns(
            (pl.col("total_equity") / pl.col("total_equity").first()).alias("cum_nav")
        )
        return daily

    # -------------------------
    # Stats
    # -------------------------
    def stats(self, daily: pl.DataFrame) -> dict[str, float]:
        if daily.is_empty():
            return {"total_return": 0.0, "ann_return": 0.0, "max_dd": 0.0, "sharpe": 0.0, "avg_hold_days": 0.0, "win_rate": 0.0}

        nav = daily["cum_nav"].fill_nan(1.0).clip(1e-8, 1e12)
        total_return = float(nav[-1]) - 1.0
        peak = nav.cum_max()
        max_dd = float(((nav / peak) - 1.0).min())

        n_days = len(daily)
        years = n_days / 252.0 if n_days else 0.0
        ann_return = (float(nav[-1])) ** (1.0 / years) - 1.0 if years > 0 else total_return

        ret = daily["strategy_return"].fill_nan(0.0)
        mu = float(ret.mean())
        sigma = float(ret.std()) if ret.len() > 1 else 0.0
        sharpe = (mu / sigma * math.sqrt(252.0)) if sigma > 1e-12 else 0.0

        if self.closed_trades:
            hold_days = [int(t["holding_days"]) for t in self.closed_trades if t.get("holding_days") is not None]
            avg_hold_days = float(np.mean(hold_days)) if hold_days else 0.0
            pnls = [float(t["pnl"]) for t in self.closed_trades]
            win_rate = float(np.mean([1.0 if p > 0 else 0.0 for p in pnls])) if pnls else 0.0
        else:
            avg_hold_days = 0.0
            win_rate = 0.0

        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "avg_hold_days": avg_hold_days,
            "win_rate": win_rate,
        }


if __name__ == "__main__":
    cfg = BacktestConfig(
        duckdb_path=os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb"),
        start_date=os.getenv("START_DATE", "20240101"),
        end_date=None,
        initial_capital=10_000_000,
        max_pair_positions=50,
        slot_capital=200_000.0,
        formation_window=250,
        top_n_liquid=500,
        corr_th=0.85,
        coint_p_th=0.05,
        top_k_pairs=100,
        liquidity_metric="amount",
        max_workers=None,
        allow_shorting=False,
    )

    bt = DynamicPairsBacktester(cfg)
    daily = bt.run()
    s = bt.stats(daily)

    print("=== 动态配对交易回测结果（Monthly pairs + Daily monitor, T+1）===")
    if not daily.is_empty():
        print(f"区间:         {daily['trade_date'][0]} ~ {daily['trade_date'][-1]}")
    print(f"总收益率:     {s['total_return']:.4f}")
    print(f"年化收益率:   {s['ann_return']:.4f}")
    print(f"最大回撤:     {s['max_dd']:.4f}")
    print(f"夏普比率:     {s['sharpe']:.4f}")
    print(f"平均持仓天数: {s['avg_hold_days']:.2f}")
    print(f"胜率(按交易): {s['win_rate']:.4f}")

    plot_nav(daily, path=pp.docs_plot_path("pairs_nav", daily, cfg.start_date, cfg.end_date))

