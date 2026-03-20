from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb
import polars as pl


@dataclass(frozen=True)
class EngineConfig:
    # ---- data / io ----
    duckdb_path: str = "shiming_daily_base.duckdb"

    # ---- adjustment (前复权) ----
    # adj_price = raw_price * adj_factor / last_adj_factor_per_stock
    adj_factor_col: str = "adj_factor"

    # ---- moving averages ----
    ma_fast: int = 5
    ma_20: int = 20
    ma_60: int = 60
    ma_250: int = 250

    # ---- bollinger ----
    boll_window: int = 20
    boll_k: float = 2.0
    boll_squeeze_threshold: float = 0.05

    # ---- rsi ----
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_oversold_days: int = 3

    # ---- gaps ----
    gap_up_ratio: float = 1.02

    # ---- volume ----
    vol_window_60: int = 60
    vol_window_20: int = 20
    bottom_volume_multiple: float = 3.0
    pullback_volume_ratio: float = 0.5
    pullback_to_ma20_tol: float = 0.015

    # ---- valuation / size ----
    davis_pe_ttm_max: float = 20.0
    davis_high_window: int = 60
    small_total_mv_max: float = 500000.0  # 单位与 tushare daily_basic 一致（通常：万元）
    small_pb_max: float = 1.2

    # ---- ROE ----
    roe_min: float = 15.0
    roe_undervalue_vs_ma250: float = 0.10  # 跌破 250 日均线超过 10%

    # ---- turnover ----
    turnover_min: float = 15.0
    rising_days: int = 3

    # ---- shiming oversold (世明超跌因子) ----
    shiming_stoch_window: int = 28
    shiming_sma1_n: int = 4  # TDX: SMA(X,4,1) ~ EMA(alpha=1/4)
    shiming_sma2_n: int = 2  # TDX: SMA(X,2,1) ~ EMA(alpha=1/2)
    shiming_x4_max: float = 11.0
    shiming_x5_max: float = 8.5
    shiming_quantile_window: int = 1000
    shiming_quantile_p: float = 0.05


class FireControlEngine:
    """
    火控推进器：从 DuckDB 装载 A 股数据，生成复权OHLC与 10 大雷达信号（向量化、LazyFrame）。
    """

    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()

    # ----------------------------
    # Data loading (DuckDB -> Polars)
    # ----------------------------
    def _read_table(self, con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> pl.DataFrame:
        cols = ", ".join(columns)
        return pl.from_arrow(con.execute(f"SELECT {cols} FROM {table}").arrow())

    def _try_has_table(self, con: duckdb.DuckDBPyConnection, table: str) -> bool:
        q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
        try:
            return bool(con.execute(q, [table]).fetchone()[0])
        except Exception:
            return False

    def load(self) -> pl.LazyFrame:
        """
        返回包含复权OHLC与信号列的 LazyFrame。

        表需求：
        - daily_data(ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
        - daily_basic(ts_code, trade_date, turnover_rate, volume_ratio, pe, pe_ttm, pb, total_mv, circ_mv)
        - adj_factor(ts_code, trade_date, adj_factor)
        可选：
        - fina_indicator(ts_code, end_date, roe)  (若存在，将按 asof 合并到 trade_date)
        """
        con = duckdb.connect(self.cfg.duckdb_path, read_only=True)
        try:
            daily_data = self._read_table(
                con,
                "daily_data",
                [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                ],
            ).lazy()

            daily_basic = self._read_table(
                con,
                "daily_basic",
                [
                    "ts_code",
                    "trade_date",
                    "turnover_rate",
                    "volume_ratio",
                    "pe",
                    "pe_ttm",
                    "pb",
                    "total_mv",
                    "circ_mv",
                ],
            ).lazy()

            adj_factor = self._read_table(
                con,
                "adj_factor",
                ["ts_code", "trade_date", self.cfg.adj_factor_col],
            ).lazy()

            fina_exists = self._try_has_table(con, "fina_indicator")
            if fina_exists:
                fina_indicator = self._read_table(
                    con,
                    "fina_indicator",
                    ["ts_code", "end_date", "roe"],
                ).lazy()
            else:
                fina_indicator = None
        finally:
            con.close()

        lf = (
            daily_data.join(daily_basic, on=["ts_code", "trade_date"], how="inner")
            .join(adj_factor, on=["ts_code", "trade_date"], how="inner")
            .with_columns(
                pl.col("trade_date").cast(pl.Utf8),
                pl.col("ts_code").cast(pl.Utf8),
                pl.col(self.cfg.adj_factor_col).cast(pl.Float64),
            )
        )

        # 可选：把 ROE asof 合并到每个交易日（取最近一期 end_date<=trade_date）
        if fina_indicator is not None:
            lf = (
                lf.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("_td"))
                .join_asof(
                    fina_indicator.with_columns(
                        pl.col("end_date").str.strptime(pl.Date, "%Y%m%d").alias("_ed")
                    ).select(["ts_code", "_ed", "roe"]),
                    left_on="_td",
                    right_on="_ed",
                    by="ts_code",
                    strategy="backward",
                )
                .drop(["_td", "_ed"])
            )
        else:
            lf = lf.with_columns(pl.lit(None, dtype=pl.Float64).alias("roe"))

        # 统一排序（rolling/shift 依赖稳定顺序）
        lf = lf.sort(["ts_code", "trade_date"])

        return self._build_signals(self._add_adjusted_ohlc(lf))

    # ----------------------------
    # Core: adjustment first
    # ----------------------------
    def _add_adjusted_ohlc(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        af = pl.col(self.cfg.adj_factor_col)
        last_af = af.last().over("ts_code")
        ratio = (af / last_af).alias("_adj_ratio")

        return (
            lf.with_columns(ratio)
            .with_columns(
                (pl.col("close") * pl.col("_adj_ratio")).alias("adj_close"),
                (pl.col("open") * pl.col("_adj_ratio")).alias("adj_open"),
                (pl.col("high") * pl.col("_adj_ratio")).alias("adj_high"),
                (pl.col("low") * pl.col("_adj_ratio")).alias("adj_low"),
            )
            .drop(["_adj_ratio"])
        )

    # ----------------------------
    # Indicators & signals (all vectorized, LazyFrame)
    # ----------------------------
    def _rsi_expr(self, close: pl.Expr) -> pl.Expr:
        # RSI = 100 - 100/(1+RS); RS = avg_gain/avg_loss (rolling mean)
        diff = close.diff().over("ts_code")
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        avg_gain = gain.rolling_mean(self.cfg.rsi_window).over("ts_code")
        avg_loss = loss.rolling_mean(self.cfg.rsi_window).over("ts_code")
        rs = avg_gain / avg_loss
        return (100.0 - (100.0 / (1.0 + rs))).alias(f"rsi_{self.cfg.rsi_window}")

    def _ema_alpha_expr(self, x: pl.Expr, n: int) -> pl.Expr:
        """
        TDX SMA(x, N, 1) 等价于 EMA(alpha=1/N)（adjust=False）。
        这里用 ewm_mean 实现，并且要求在外部已经按 ts_code, trade_date 排序。
        """
        alpha = 1.0 / float(n)
        return x.ewm_mean(alpha=alpha, adjust=False, min_periods=1)

    def _build_signals(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        c = pl.col("adj_close")
        o = pl.col("adj_open")
        h = pl.col("adj_high")
        l = pl.col("adj_low")
        vol = pl.col("vol")

        ma20 = c.rolling_mean(self.cfg.ma_20).over("ts_code").alias("ma20")
        ma60 = c.rolling_mean(self.cfg.ma_60).over("ts_code").alias("ma60")
        ma5 = c.rolling_mean(self.cfg.ma_fast).over("ts_code").alias("ma5")
        ma250 = c.rolling_mean(self.cfg.ma_250).over("ts_code").alias("ma250")

        # Bollinger (window=boll_window)
        mid = c.rolling_mean(self.cfg.boll_window).over("ts_code").alias("boll_mid")
        std = c.rolling_std(self.cfg.boll_window).over("ts_code").alias("boll_std")
        upper = (pl.col("boll_mid") + pl.lit(self.cfg.boll_k) * pl.col("boll_std")).alias("boll_upper")
        lower = (pl.col("boll_mid") - pl.lit(self.cfg.boll_k) * pl.col("boll_std")).alias("boll_lower")

        # RSI
        rsi = self._rsi_expr(c)

        # rolling helpers
        vol_ma60 = vol.rolling_mean(self.cfg.vol_window_60).over("ts_code").alias("vol_ma60")
        vol_ma20 = vol.rolling_mean(self.cfg.vol_window_20).over("ts_code").alias("vol_ma20")
        high_60 = c.rolling_max(self.cfg.davis_high_window).over("ts_code").alias("high_60")

        # shifts
        ma20_prev = pl.col("ma20").shift(1).over("ts_code")
        close_prev = c.shift(1).over("ts_code")
        high_prev = h.shift(1).over("ts_code")

        # 10 signals (bool; you can cast to Int8 downstream if needed)
        sig_ma_bull = (
            (c > pl.col("ma20")) & (pl.col("ma20") > pl.col("ma60")) & (pl.col("ma20") > ma20_prev)
        ).alias("sig_ma_bull")

        sig_boll_squeeze_break = (
            ((pl.col("boll_upper") - pl.col("boll_lower")) / pl.col("boll_mid") < pl.lit(self.cfg.boll_squeeze_threshold))
            & (c > pl.col("boll_upper"))
        ).alias("sig_boll_squeeze_break")

        # RSI oversold 3 days then cross above oversold today
        rsi_col = pl.col(f"rsi_{self.cfg.rsi_window}")
        below = (rsi_col < pl.lit(self.cfg.rsi_oversold)).cast(pl.Int8)
        below_3 = (below.rolling_sum(self.cfg.rsi_oversold_days).over("ts_code") == self.cfg.rsi_oversold_days)
        sig_rsi_oversold_reversal = (below_3.shift(1).over("ts_code") & (rsi_col > pl.lit(self.cfg.rsi_oversold))).alias(
            "sig_rsi_oversold_reversal"
        )

        sig_gap_up = ((o > high_prev * pl.lit(self.cfg.gap_up_ratio)) & (l >= high_prev)).alias("sig_gap_up")

        sig_bottom_volume = (
            (vol > pl.col("vol_ma60") * pl.lit(self.cfg.bottom_volume_multiple)) & (c > o)
        ).alias("sig_bottom_volume")

        sig_pullback_low_volume = (
            (vol < pl.col("vol_ma20") * pl.lit(self.cfg.pullback_volume_ratio))
            & ((c - pl.col("ma20")).abs() / pl.col("ma20") < pl.lit(self.cfg.pullback_to_ma20_tol))
        ).alias("sig_pullback_low_volume")

        sig_davis_double = ((pl.col("pe_ttm") < pl.lit(self.cfg.davis_pe_ttm_max)) & (c >= pl.col("high_60"))).alias(
            "sig_davis_double"
        )

        # MA5 golden cross MA20: today ma5>ma20 and yesterday ma5<=ma20
        ma5_prev = pl.col("ma5").shift(1).over("ts_code")
        sig_smallcap_reversal = (
            (pl.col("total_mv") < pl.lit(self.cfg.small_total_mv_max))
            & (pl.col("pb") < pl.lit(self.cfg.small_pb_max))
            & (pl.col("ma5") > pl.col("ma20"))
            & (ma5_prev <= ma20_prev)
        ).alias("sig_smallcap_reversal")

        sig_high_roe_misprice = (
            (pl.col("roe") > pl.lit(self.cfg.roe_min))
            & (c < pl.col("ma250") * (1.0 - pl.lit(self.cfg.roe_undervalue_vs_ma250)))
        ).alias("sig_high_roe_misprice")

        # 3-day rising closes + high turnover today
        rising_3 = (
            (c > close_prev)
            & (close_prev > c.shift(2).over("ts_code"))
            & (c.shift(2).over("ts_code") > c.shift(3).over("ts_code"))
        )
        sig_high_turnover_follow = ((pl.col("turnover_rate") > pl.lit(self.cfg.turnover_min)) & rising_3).alias(
            "sig_high_turnover_follow"
        )

        # ----------------------------
        # 11) 世明私募基金自创核心超跌因子（来自通达信脚本）
        # X3 := (CLOSE-LLV(LOW,28)) / (HHV(HIGH,28)-LLV(LOW,28)) * 100
        # X4 := SMA(X3,4,1)  ~ EMA(alpha=1/4)
        # X5 := SMA(X4,2,1)  ~ EMA(alpha=1/2)
        #
        # 因子：factor_shiming_oversold = X5 （数值越低越“超跌”）
        # 触发：X5 & X4 同时处于通达信阈值区间，并且
        #      factor 处于过去 1000 日低位 5% 分位点，且今日收盘价站上 5 日均线
        # 全程只使用 T 日及之前数据（无未来函数）。
        # ----------------------------
        llv = l.rolling_min(self.cfg.shiming_stoch_window).over("ts_code")
        hhv = h.rolling_max(self.cfg.shiming_stoch_window).over("ts_code")
        denom = (hhv - llv).replace(0.0, None)

        # 注意：同一个 with_columns 批次里不能引用“刚创建”的列名
        # 所以这里全部用 Expr 串联，避免 _sh_x3/_sh_x4 依赖顺序问题
        x3_expr = (((c - llv) / denom) * 100.0).fill_nan(None)
        x4_expr = self._ema_alpha_expr(x3_expr, self.cfg.shiming_sma1_n).over("ts_code")
        x5_expr = self._ema_alpha_expr(x4_expr, self.cfg.shiming_sma2_n).over("ts_code")

        x4 = x4_expr.alias("_sh_x4")
        x5 = x5_expr.alias("factor_shiming_oversold")

        sh_q = (
            x5_expr.rolling_quantile(self.cfg.shiming_quantile_p, window_size=self.cfg.shiming_quantile_window)
            .over("ts_code")
            .alias("_sh_q")
        )
        sh_core = (
            (x5_expr < pl.lit(self.cfg.shiming_x5_max)) & (x4_expr < pl.lit(self.cfg.shiming_x4_max))
        ).alias("_sh_core")

        return (
            lf.with_columns([ma20, ma60, ma5, ma250, mid, std])
            .with_columns([upper, lower, rsi, vol_ma60, vol_ma20, high_60])
            .with_columns(
                [
                    sig_ma_bull,
                    sig_boll_squeeze_break,
                    sig_rsi_oversold_reversal,
                    sig_gap_up,
                    sig_bottom_volume,
                    sig_pullback_low_volume,
                    sig_davis_double,
                    sig_smallcap_reversal,
                    sig_high_roe_misprice,
                    sig_high_turnover_follow,
                    # 11th indicator
                    x4,
                    x5,
                    sh_q,
                    sh_core,
                ]
            )
            .with_columns(
                (
                    pl.col("_sh_core")
                    & (pl.col("factor_shiming_oversold") <= pl.col("_sh_q"))
                    & (c > pl.col("ma5"))
                ).alias("sig_shiming_oversold")
            )
            .drop(["_sh_x4", "_sh_q", "_sh_core"])
        )

    # ----------------------------
    # Convenience
    # ----------------------------
    def scan(self) -> pl.LazyFrame:
        """别名：load()"""
        return self.load()

    def collect(self) -> pl.DataFrame:
        """一次性计算并返回 DataFrame（大库建议先 filter 再 collect）"""
        return self.load().collect()


__all__ = ["EngineConfig", "FireControlEngine"]

