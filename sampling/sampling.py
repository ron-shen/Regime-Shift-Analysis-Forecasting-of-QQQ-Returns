from __future__ import annotations

from pathlib import Path
from typing import Literal
import pandas as pd


COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def _read_mt_csv(path: Path) -> pd.DataFrame:
    """
    Reads the MT-style file that is often TAB-separated (even if extension is .csv).
    Example row:
      2010.01.03    17:00   1.6109  1.6109  1.6105  1.6105  0

    We support either tab or comma delimiter.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        sep=r"[\t,]+",
        engine="python",
        dtype={
            "date": "string",
            "time": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    return df


def minute_to_toronto_session_daily_ohlc(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    *,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> pd.DataFrame:
    """
    Input df columns: date, time, open, high, low, close, volume
    - date format: YYYY.MM.DD
    - time format: HH:MM
    Timestamps are in FIXED EST (UTC-5) with NO DST adjustment.

    Filters bars to a Toronto-local session time window, then aggregates into daily OHLC.

    Args:
        session_start/session_end:
            "HH:MM" (24h). If session_end < session_start, the window is treated as crossing midnight.
        start_inclusive/end_inclusive:
            Control boundary inclusivity.

    Output index: session_date (Toronto date)
    Output cols: open, high, low, close, volume, n_bars
    """
    # Parse naive datetime
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M", errors="raise")

    # Step 1: localize as FIXED UTC-5 (no DST). Use a fixed-offset tz.
    # NOTE: "Etc/GMT+5" is UTC-5 (yes, the sign is inverted in Etc/GMT zones).
    ts_fixed_est = ts.dt.tz_localize("Etc/GMT+5")

    # Step 2: convert to Toronto (DST-aware)
    ts_tor = ts_fixed_est.dt.tz_convert("America/Toronto")

    out = df.copy()
    out["ts_toronto"] = ts_tor

    # ---- Session filtering (configurable) ----
    # Use time-of-day from tz-aware timestamps (not string compare)
    tor_time = out["ts_toronto"].dt.time
    start_t = pd.to_datetime(session_start, format="%H:%M").time()
    end_t = pd.to_datetime(session_end, format="%H:%M").time()

    def ge(a, b):  # a >= b (optionally strict)
        return a >= b if start_inclusive else a > b

    def le(a, b):  # a <= b (optionally strict)
        return a <= b if end_inclusive else a < b

    if end_t >= start_t:
        # Normal same-day window, e.g. 09:30–16:00
        mask = ge(tor_time, start_t) & le(tor_time, end_t)
    else:
        # Cross-midnight window, e.g. 20:00–02:00
        mask = ge(tor_time, start_t) | le(tor_time, end_t)

    out = out.loc[mask].copy()
    # -----------------------------------------

    if out.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "n_bars"]).astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "n_bars": "int64",
            }
        )

    # Group by Toronto calendar date (session date)
    out["session_date"] = out["ts_toronto"].dt.tz_localize(None).dt.date

    # Daily OHLC from minute bars within the session
    g = out.groupby("session_date", sort=True)
    daily = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "n_bars": g.size(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "session_date"
    return daily


def process_year_file(
    path: Path,
    session_start: str = "09:30",
    session_end: str = "16:00",
    *,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> pd.DataFrame:
    df = _read_mt_csv(path)
    return minute_to_toronto_session_daily_ohlc(
        df,
        session_start=session_start,
        session_end=session_end,
        start_inclusive=start_inclusive,
        end_inclusive=end_inclusive,
    )


def process_all_years(
    folder: str | Path,
    symbol: str = "GBPUSD",
    start_year: int = 2010,
    end_year: int = 2025,
    *,
    session_start: str = "09:30",
    session_end: str = "16:00",
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> pd.DataFrame:
    folder = Path(folder)

    all_daily = []
    for year in range(start_year, end_year + 1):
        fname = f"DAT_MT_{symbol}_M1_{year}.csv"
        fpath = folder / fname
        if not fpath.exists():
            continue

        daily = process_year_file(
            fpath,
            session_start=session_start,
            session_end=session_end,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
        )
        if not daily.empty:
            daily["year"] = year
            all_daily.append(daily)

    if not all_daily:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "n_bars", "year"])

    return pd.concat(all_daily).sort_index()


def save_outputs(daily: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=True)


if __name__ == "__main__":
    # Example: regular session
    daily = process_all_years(
        "./data/GBPUSDM1",
        symbol="GBPUSD",
        start_year=2010,
        end_year=2025,
        session_start="09:30",
        session_end="16:00",
    )

    # Example: cross-midnight window (Toronto local time)
    # daily_night = process_all_years(
    #     "./data/GBPUSDM1",
    #     symbol="GBPUSD",
    #     start_year=2010,
    #     end_year=2025,
    #     session_start="20:00",
    #     session_end="02:00",
    #     end_inclusive=False,
    # )

    pass