"""
从 Tushare 增量更新 A 股日频数据、财务数据到 DuckDB。
- 配置：.env 中 TUSHARE_TOKEN、DUCKDB_PATH（可选 START_DATE，默认 20150101）
- 日频：按股票拉取，避免单次 5000 行截断；每次运行只补缺失区间，快速更新
- 财务：按股票拉取，每次只补该股缺失的 end_date 区间
- 限流：单次调用间隔 + 命中限流时退避重试
"""
import os
import time
import warnings
from datetime import datetime

import duckdb
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

warnings.simplefilter("ignore", category=FutureWarning)
load_dotenv()

# ---------- 配置（.env 优先） ----------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN") or ""
DB_PATH = os.getenv("DUCKDB_PATH", "shiming_daily_base.duckdb")
START_DATE = os.getenv("START_DATE", "20150101")
END_DATE = os.getenv("END_DATE")  # 不设则用今天
DAILY_SLEEP = float(os.getenv("DAILY_SLEEP_SEC", "0.3"))  # 日频每只股后休眠，避免限流
FIN_SLEEP = float(os.getenv("FIN_SLEEP_SEC", "0.35"))  # 财务 200 次/分钟，约 0.3~0.35
RETRY_TIMES = int(os.getenv("RETRY_TIMES", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF_SEC", "2.0"))
RATE_LIMIT_WAIT = float(os.getenv("RATE_LIMIT_WAIT_SEC", "35"))  # 命中限流时等待秒数

if not TUSHARE_TOKEN:
    raise RuntimeError("未配置 TUSHARE_TOKEN，请在 .env 中设置")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def _retry(fn, label: str):
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "最多访问该接口" in msg or "每分钟最多访问" in msg:
                print(f"⏳ {label} 命中限流，等待 {RATE_LIMIT_WAIT:.0f}s 后重试")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            if attempt >= RETRY_TIMES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)
            print(f"⚠️ {label} 第 {attempt}/{RETRY_TIMES} 次失败: {e}")


def _end_date() -> str:
    return END_DATE or datetime.now().strftime("%Y%m%d")


def init_db(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_data (
            ts_code VARCHAR, trade_date VARCHAR, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, pre_close DOUBLE, change DOUBLE,
            pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code VARCHAR, trade_date VARCHAR, turnover_rate DOUBLE,
            volume_ratio DOUBLE, pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE,
            total_mv DOUBLE, circ_mv DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fina_indicator (
            ts_code VARCHAR, ann_date VARCHAR, end_date VARCHAR,
            eps DOUBLE, dt_eps DOUBLE, gross_margin DOUBLE,
            roe DOUBLE, roa DOUBLE, debt_to_assets DOUBLE,
            PRIMARY KEY (ts_code, end_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS adj_factor (
            ts_code VARCHAR, trade_date VARCHAR, adj_factor DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    print(f"✅ DuckDB 表已就绪 -> {os.path.abspath(DB_PATH)}")


def get_stock_list() -> list[str]:
    df = _retry(lambda: pro.stock_basic(list_status="L", fields="ts_code"), "stock_basic")
    if df is None or df.empty:
        return []
    return df["ts_code"].astype(str).tolist()


def get_last_trade_date(con: duckdb.DuckDBPyConnection, ts_code: str) -> str | None:
    try:
        r = con.execute(
            "SELECT max(trade_date) FROM daily_data WHERE ts_code = ?",
            [ts_code],
        ).fetchone()
        return r[0] if r and r[0] else None
    except Exception:
        return None


def get_last_fin_end_date(con: duckdb.DuckDBPyConnection, ts_code: str) -> str | None:
    try:
        r = con.execute(
            "SELECT max(end_date) FROM fina_indicator WHERE ts_code = ?",
            [ts_code],
        ).fetchone()
        return r[0] if r and r[0] else None
    except Exception:
        return None


def _next_day(date_str: str) -> str:
    t = pd.Timestamp(date_str)
    return (t + pd.Timedelta(days=1)).strftime("%Y%m%d")


def sync_daily_incremental(con: duckdb.DuckDBPyConnection):
    """按股票增量同步日频：日线、估值、复权因子。每只股只拉「最后日期+1 ~ 今天」避免截断。"""
    stocks = get_stock_list()
    if not stocks:
        print("⚠️ 未获取到股票列表，跳过日频同步")
        return
    end = _end_date()
    total_daily, total_basic, total_adj = 0, 0, 0
    for i, code in enumerate(stocks, 1):
        try:
            last = get_last_trade_date(con, code)
            start = _next_day(last) if last else START_DATE
            if start > end:
                continue
            # 日线
            df_d = _retry(
                lambda c=code: pro.daily(ts_code=c, start_date=start, end_date=end),
                f"daily {code}",
            )
            if df_d is not None and not df_d.empty:
                con.register("_d", df_d)
                con.execute("""
                    INSERT OR IGNORE INTO daily_data
                    SELECT ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
                    FROM _d
                """)
                total_daily += len(df_d)
            # 估值
            df_b = _retry(
                lambda c=code: pro.daily_basic(
                    ts_code=c, start_date=start, end_date=end,
                    fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,total_mv,circ_mv",
                ),
                f"daily_basic {code}",
            )
            if df_b is not None and not df_b.empty:
                con.register("_b", df_b)
                con.execute("""
                    INSERT OR IGNORE INTO daily_basic
                    SELECT ts_code, trade_date, turnover_rate, volume_ratio, pe, pe_ttm, pb, total_mv, circ_mv
                    FROM _b
                """)
                total_basic += len(df_b)
            # 复权因子
            df_a = _retry(
                lambda c=code: pro.adj_factor(ts_code=c, start_date=start, end_date=end),
                f"adj_factor {code}",
            )
            if df_a is not None and not df_a.empty:
                con.register("_a", df_a)
                con.execute("""
                    INSERT OR IGNORE INTO adj_factor
                    SELECT ts_code, trade_date, adj_factor FROM _a
                """)
                total_adj += len(df_a)
        except Exception as e:
            print(f"⚠️ {code} 日频异常: {e}")
        if i % 100 == 0:
            print(f"📌 日频进度 {i}/{len(stocks)}，本轮已入库 日线={total_daily} 估值={total_basic} 复权={total_adj}")
        time.sleep(DAILY_SLEEP)
    print(f"✅ 日频增量同步完成 -> 日线 {total_daily} 估值 {total_basic} 复权 {total_adj}")


def sync_finance_incremental(con: duckdb.DuckDBPyConnection):
    """按股票增量同步财务：只拉该股 end_date 在「最后期末+1 天 ~ 今天」的区间。"""
    stocks = get_stock_list()
    if not stocks:
        print("⚠️ 未获取到股票列表，跳过财务同步")
        return
    end = _end_date()
    total = 0
    for i, code in enumerate(stocks, 1):
        try:
            last = get_last_fin_end_date(con, code)
            start = _next_day(last) if last else START_DATE
            if start > end:
                continue
            df = _retry(
                lambda c=code: pro.fina_indicator(
                    ts_code=c, start_date=start, end_date=end,
                    fields="ts_code,ann_date,end_date,eps,dt_eps,gross_margin,roe,roa,debt_to_assets",
                ),
                f"fina {code}",
            )
            if df is not None and not df.empty:
                con.register("_f", df)
                con.execute("""
                    INSERT OR IGNORE INTO fina_indicator
                    SELECT ts_code, ann_date, end_date, eps, dt_eps, gross_margin, roe, roa, debt_to_assets
                    FROM _f
                """)
                total += len(df)
        except Exception as e:
            print(f"⚠️ {code} 财务异常: {e}")
        if i % 100 == 0:
            print(f"📌 财务进度 {i}/{len(stocks)}，本轮已入库 {total}")
        time.sleep(FIN_SLEEP)
    print(f"✅ 财务增量同步完成 -> 本轮回库 {total} 行")


def main():
    con = duckdb.connect(DB_PATH)
    try:
        init_db(con)
        print("\n--- 🚀 日频增量更新 ---")
        sync_daily_incremental(con)
        print("\n--- 🚀 财务增量更新 ---")
        sync_finance_incremental(con)
    finally:
        con.close()
    print("\n🏁 更新结束，连接已关闭。")


if __name__ == "__main__":
    main()
