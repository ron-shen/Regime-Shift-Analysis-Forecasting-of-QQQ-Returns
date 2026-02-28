#!/usr/bin/env python3
"""
pipeline_feature_engineering.py

Thin orchestrator that glues:
- data_loaders.py
- feature_apply.py

Builds:
- X: engineered features + optional passthrough raw columns
- y: label outputs (fwd_ret / fwd_ret_label) unless allow_labels=True
- raw: all raw columns (prefixed) for sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, List

import pandas as pd
from features.ta import intermarket_divergence
from data_loader.data_loaders import load_daily_csv, load_m1_folder_to_daily
from features.apply import apply_feature_args, prefix_cols
from features.ta import beta

@dataclass(frozen=True)
class DatasetSpec:
    """
    One dataset = one spec.

    If you want *no transforms*, set features=[] and choose passthrough_cols/include_all_raw_in_X.
    """
    name: str
    kind: str                 # "daily_csv" or "m1_folder"
    source: str | Path
    features: Sequence[str]   # e.g. ["ema:20", "roc:10"]

    # --- Daily CSV options ---
    date_col: str = "Date"
    dayfirst: bool = False
    value_col: Optional[str] = None   # used to create Close for transforms on non-OHLC datasets

    # --- M1 folder options ---
    symbol: str = ""
    start_year: int = 2010
    end_year: int = 2025
    session_start: str = "09:30"
    session_end: str = "16:00"

    # --- Passthrough ---
    passthrough_cols: Optional[Sequence[str]] = None
    include_all_raw_in_X: bool = False


def build_feature_table(
    specs: Sequence[DatasetSpec],
    *,
    allow_labels: bool = False,
    join: str = "outer",
    drop_all_nan_rows: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build:
      X = engineered features + optional passthrough cols
      y = label outputs
      raw = all raw columns (prefixed)

    join="outer" keeps all dates; join="inner" keeps only overlapping dates.
    """
    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.DataFrame] = []
    raw_parts: List[pd.DataFrame] = []

    for spec in specs:
        # ---- Load to daily ----
        if spec.kind == "daily_csv":
            df = load_daily_csv(
                spec.source,
                dataset_name=spec.name,
                date_col=spec.date_col,
                dayfirst=spec.dayfirst,
                ensure_close_col=bool(spec.features),  # only require Close if we will run transforms
                value_col=spec.value_col,
            )
        elif spec.kind == "m1_folder":
            if not spec.symbol:
                raise ValueError(f"{spec.name}: symbol is required for kind='m1_folder'")
            df = load_m1_folder_to_daily(
                spec.source,
                dataset_name=spec.name,
                symbol=spec.symbol,
                start_year=spec.start_year,
                end_year=spec.end_year,
                session_start=spec.session_start,
                session_end=spec.session_end,
            )
        else:
            raise ValueError(f"{spec.name}: unknown kind '{spec.kind}'")

        if df.empty:
            continue

        # ---- Raw (always kept for checks) ----
        raw_parts.append(prefix_cols(df, spec.name))

        # ---- Passthrough into X (optional) ----
        if spec.include_all_raw_in_X:
            X_parts.append(prefix_cols(df, spec.name))
        elif spec.passthrough_cols:
            missing = [c for c in spec.passthrough_cols if c not in df.columns]
            if missing:
                raise KeyError(f"{spec.name}: passthrough_cols missing: {missing}. Columns: {list(df.columns)}")
            X_parts.append(prefix_cols(df.loc[:, list(spec.passthrough_cols)], spec.name))

        # ---- Engineered features ----
        X_i, y_i = apply_feature_args(df, spec.features, dataset_prefix=spec.name, allow_labels=allow_labels)
        if not X_i.empty:
            X_parts.append(X_i)
        if not y_i.empty:
            y_parts.append(y_i)

    X = pd.concat(X_parts, axis=1, join=join).sort_index() if X_parts else pd.DataFrame()
    y = pd.concat(y_parts, axis=1, join=join).sort_index() if y_parts else pd.DataFrame()
    raw = pd.concat(raw_parts, axis=1, join=join).sort_index() if raw_parts else pd.DataFrame()

    if drop_all_nan_rows and not X.empty:
        X = X.dropna(how="all")

    return {"X": X, "y": y, "raw": raw}


if __name__ == "__main__":
    # Example
    specs = [
        #qqq
        DatasetSpec(
            name="qqq",
            kind="daily_csv",
            source="./data/qqq/qqq.csv",
            date_col="Date",
            features = [
                "rsi:14",
                "macd:12:26:9",

                "roc:1", "roc:2", "roc:3", "roc:4", "roc:5",
                "roc:6", "roc:7", "roc:8", "roc:9", "roc:10",
                "roc:11", "roc:12", "roc:13", "roc:14", "roc:15",
                "roc:16", "roc:17", "roc:18", "roc:19", "roc:20",
                "roc:21",

                "volatility:20",
                "obv:20",
                "fwd_ret_label:1",
                "fwd_ret:1"
            ],
            passthrough_cols = ['Close', 'closing_bid', 'closing_ask']
        ),
        #DIX
        DatasetSpec(
            name="dix",
            kind="daily_csv",
            source="./data/dark-index/dix.csv",
            date_col="Date",
            features=["macd:12:26:9"]
        ),
        #GEX
        DatasetSpec(
            name="gex",
            kind="daily_csv",
            source="./data/dark-index/gex.csv",
            date_col="Date",
            features=["roc:1"]
        ),
        #GLD
        DatasetSpec(
            name="gld",
            kind="daily_csv",
            source="./data/gld-etf/GLD.csv",
            symbol="gld",
            date_col="Date",
            features=["roc:1"],
            passthrough_cols = ['Close']
        ),
        #reserve repo
        DatasetSpec(
            name="rrp",
            kind="daily_csv",
            source="./data/rrp/RRPONTSYD.csv",
            date_col="observation_date",
            features=["roc:5"]
        ),
        #VIX
        DatasetSpec(
            name="vix",
            kind="daily_csv",
            source="./data/vix/vix.csv",
            date_col="Date",
            features=["roc:1"]
        ),
        #GBPUSD
        DatasetSpec(
            name="gbpusd",
            kind="m1_folder",
            source="./data/GBPUSDM1",
            symbol="gbpusd",
            start_year=2010,
            end_year=2025,
            session_start="09:30",
            session_end="16:00",
            features=["roc:1", "STOK:20"],
            passthrough_cols = ['Close']

        ),
        #USDJPY
        DatasetSpec(
            name="usdjpy",
            kind="m1_folder",
            source="./data/USDJPYM1",
            symbol="usdjpy",
            start_year=2010,
            end_year=2025,
            session_start="09:30",
            session_end="16:00",
            features=["roc:1"],
            passthrough_cols= ['Close']
        ),
        #2 year treasury yield
        DatasetSpec(
            name="t2y",
            kind="daily_csv",
            source="./data/us-treasury/T2Y.csv",
            date_col="observation_date",
            features=["diff:1", "acc:1"],
            passthrough_cols=['DGS2']
        ),
        #10y-2y treasury spread
        DatasetSpec(
            name="t10y2y",
            kind="daily_csv",
            source="./data/us-treasury/T10Y2Y.csv",
            date_col="observation_date",
            features=["diff:1"]
        ),
    ]

    tmp_out = build_feature_table(specs, allow_labels=True, join="outer")
    X = tmp_out["X"]
    X.dropna(subset=['qqq_Close'], inplace=True)  # only keep rows where we have qqq_Close
    y = tmp_out["y"]
    #calculate beta separately since it needs two datasets
    X['beta_20_gld_qqq'] = beta(window=20, asset_col='gld_Close', benchmark_col='qqq_Close', df=X)
    X['beta_20_gbpusd_qqq'] = beta(window=20, asset_col='gbpusd_Close', benchmark_col='qqq_Close', df=X)
    X['beta_20_usdjpy_qqq'] = beta(window=20, asset_col='usdjpy_Close', benchmark_col='qqq_Close', df=X)
    #bid ask apread
    X['bid_ask_spread'] = X['qqq_closing_ask'] - X['qqq_closing_bid']
    #Yield-Equity Divergence
    X['qqq_us2y_corr'] = intermarket_divergence(20, X['t2y_DGS2'], X['qqq_Close'])

    res = pd.concat([X, y], axis=1)
    res.to_csv("../data/full-dataset/full_dataset.csv")
    #out["X"].to_csv("./out/features.csv")
    #print("Saved ./out/features.csv", out["X"].shape)
