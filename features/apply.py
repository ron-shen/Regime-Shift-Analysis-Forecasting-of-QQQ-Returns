"""
feature_apply.py

Feature application helper:

- apply_feature_args: run CLI-style feature strings (e.g., "ema:20") using feature_engineering_cli.py
- Splits label-like outputs (fwd_ret, fwd_ret_label) into y unless allow_labels=True
"""

from __future__ import annotations

from typing import Sequence, Tuple, List

import pandas as pd

from features.engineering import *

LABEL_FEATURES = {"fwd_ret", "fwd_ret_label"}


def prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out


def apply_feature_args(
    df: pd.DataFrame,
    feature_args: Sequence[str],
    *,
    dataset_prefix: str,
    allow_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply CLI-style feature args like ["ema:20", "rsi:14"] using FEATURE_FUNCTIONS.

    Returns:
      X_df (features), y_df (labels)
    """
    if not feature_args:
        return pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)

    if df.index.name != "Date":
        raise ValueError("apply_feature_args expects a Date-indexed daily DataFrame.")

    feat_series: List[pd.Series] = []
    label_series: List[pd.Series] = []

    for arg in feature_args:
        name, params = parse_feature_arg(arg)
        if name not in FEATURE_FUNCTIONS:
            raise ValueError(f"Unknown feature '{name}'. Available: {list(FEATURE_FUNCTIONS.keys())}")

        # Convert numeric params
        proc = []
        for p in params:
            s = str(p)
            if s.isdigit():
                proc.append(int(s))
            else:
                try:
                    proc.append(float(s))
                except ValueError:
                    proc.append(s)

        result = FEATURE_FUNCTIONS[name](*proc, df)
        bucket = label_series if (name in LABEL_FEATURES and allow_labels) else feat_series

        if isinstance(result, pd.Series):
            col = f"{name}_{proc[0]}" if proc else name
            bucket.append(result.rename(f"{dataset_prefix}_{col}"))
        elif isinstance(result, pd.DataFrame):
            tmp = prefix_cols(result, dataset_prefix)
            for c in tmp.columns:
                bucket.append(tmp[c])
        else:
            raise TypeError(f"{dataset_prefix}:{name} must return Series/DataFrame, got {type(result)}")

    X = pd.concat(feat_series, axis=1) if feat_series else pd.DataFrame(index=df.index)
    y = pd.concat(label_series, axis=1) if label_series else pd.DataFrame(index=df.index)
    return X, y
