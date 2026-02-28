"""
data_loaders.py

Loading helpers for the daily feature-engineering pipeline.

- load_daily_csv: read heterogeneous daily CSVs into a Date-indexed DataFrame
- load_m1_folder_to_daily: convert per-year M1 files into daily OHLCV via sampling.process_all_years
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from sampling.sampling import *


def standardize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common OHLCV columns to Open/High/Low/Close/Volume (title-case)."""
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "open":
            rename[c] = "Open"
        elif lc == "high":
            rename[c] = "High"
        elif lc == "low":
            rename[c] = "Low"
        elif lc == "close":
            rename[c] = "Close"
        elif lc in ("volume", "vol"):
            rename[c] = "Volume"
        elif lc in ("adj close", "adj_close", "adjclose"):
            rename[c] = "Adj Close"
    return df.rename(columns=rename)


def ensure_close(df: pd.DataFrame, *, dataset_name: str, value_col: Optional[str]) -> pd.DataFrame:
    """
    Ensure df has a 'Close' column, required by most TA indicators.
    Use `value_col` to choose which series becomes Close for non-OHLC datasets.
    """
    if "Close" in df.columns:
        return df

    if value_col:
        if value_col not in df.columns:
            raise KeyError(f"{dataset_name}: value_col='{value_col}' not found. Columns: {list(df.columns)}")
        df = df.copy()
        df["Close"] = pd.to_numeric(df[value_col], errors="coerce")
        return df

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        df = df.copy()
        df["Close"] = df[numeric_cols[0]]
        return df

    raise ValueError(
        f"{dataset_name}: no 'Close' column. Provide value_col (e.g., 'dix' or 'RRPONTSYD')."
    )


def load_daily_csv(
    path: str | Path,
    *,
    dataset_name: str,
    date_col: str,
    dayfirst: bool = False,
    ensure_close_col: bool = False,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a daily CSV into Date-indexed DF.

    - date_col can be "Date" or "observation_date" etc.
    - set dayfirst=True if your dates are dd/mm/yyyy
    - ensure_close_col=True only if you will run TA transforms on it
    """
    path = Path(path)
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise KeyError(f"{dataset_name}: date_col='{date_col}' not found in {path.name}")

    df[date_col] = pd.to_datetime(df[date_col], errors="raise", dayfirst=dayfirst)
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = standardize_ohlcv_cols(df)

    # Coerce numeric where possible (helps with blank values e.g. RRP)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    if ensure_close_col:
        df = ensure_close(df, dataset_name=dataset_name, value_col=value_col)

    return df


def load_m1_folder_to_daily(
    folder: str | Path,
    *,
    dataset_name: str,
    symbol: str,
    start_year: int,
    end_year: int,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    """
    Convert M1 per-year files into daily OHLCV using sampling.process_all_years().
    """
    daily = process_all_years(
        folder=str(folder),
        symbol=symbol,
        start_year=start_year,
        end_year=end_year,
        session_start=session_start,
        session_end=session_end,
    )
    if daily.empty:
        return daily

    daily = daily.copy()
    daily.index.name = "Date"
    daily = daily.sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    daily = standardize_ohlcv_cols(daily)

    # TA transforms need Close; M1 -> daily always has it.
    daily = ensure_close(daily, dataset_name=dataset_name, value_col=None)
    return daily
