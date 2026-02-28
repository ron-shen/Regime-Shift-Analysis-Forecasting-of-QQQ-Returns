import math
import numpy as np
import pandas as pd
import pytest

from data_loader.data_loaders import load_daily_csv, ensure_close
from features.apply import apply_feature_args
from pipeline.feature_pipeline import DatasetSpec, build_feature_table

import features.engineering as fe_cli  # holds FEATURE_FUNCTIONS + parse_feature_arg
import features.ta as ta_mod           # your indicator implementations
import utils.utility as util_mod       # shift()


# -----------------------
# Helpers / fixtures
# -----------------------
def make_ohlcv_df(start="2020-01-01", n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    # Create a smooth-ish price series with noise
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    open_ = close + rng.normal(0, 0.2, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, size=n))
    vol = rng.integers(100, 500, size=n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )
    df.index.name = "Date"
    return df


def write_csv(df: pd.DataFrame, path):
    # Ensure Date is a column for load_daily_csv(date_col=...)
    out = df.reset_index()
    out.to_csv(path, index=False)


# -----------------------
# 1) Loader-level correctness
# -----------------------
def test_ensure_close_uses_value_col():
    df = pd.DataFrame(
        {"dix": [1.0, 2.5, 3.2]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    df.index.name = "Date"

    out = ensure_close(df, dataset_name="dark", value_col="dix")
    assert "Close" in out.columns
    assert np.allclose(out["Close"].values, df["dix"].values)


# -----------------------
# 3) Alignment across specs / join logic
# -----------------------
def test_build_feature_table_outer_join_aligns_indices(tmp_path):
    # dataset A: 2020-01-01 .. 2020-01-10
    df_a = make_ohlcv_df(start="2020-01-01", n=10, seed=4)
    p_a = tmp_path / "a.csv"
    write_csv(df_a, p_a)

    # dataset B: 2020-01-06 .. 2020-01-15 (overlap is 5 days)
    df_b = make_ohlcv_df(start="2020-01-06", n=10, seed=5)
    p_b = tmp_path / "b.csv"
    write_csv(df_b, p_b)

    specs = [
        DatasetSpec(
            name="a",
            kind="daily_csv",
            source=p_a,
            date_col="Date",
            features=["roc:1", "weekday"],
        ),
        DatasetSpec(
            name="b",
            kind="daily_csv",
            source=p_b,
            date_col="Date",
            features=["roc:1", "month"],
        ),
    ]

    out = build_feature_table(specs, allow_labels=True, join="outer")
    X, raw = out["X"], out["raw"]

    # union of dates = 2020-01-01 .. 2020-01-15
    exp_idx = pd.date_range("2020-01-01", "2020-01-15", freq="D")
    assert (X.index == exp_idx).all()

    # column prefixing must exist
    assert any(c.startswith("a__") for c in X.columns)
    assert any(c.startswith("b__") for c in X.columns)

    # raw must keep all raw columns with prefixes
    assert any(c.startswith("a__Close") for c in raw.columns)
    assert any(c.startswith("b__Close") for c in raw.columns)

    # Missing region should be NaN for non-overlap
    # For example: b__month shouldn't exist before 2020-01-06
    b_month_cols = [c for c in X.columns if c.startswith("b__") and "month" in c]
    assert b_month_cols, "Expected b__month feature column(s)"
    assert X.loc[pd.Timestamp("2020-01-03"), b_month_cols].isna().all()


# -----------------------
# 4) No leakage tests (causality / perturb-future invariance)
# -----------------------
def _compute_feature(df, feature_arg, prefix="ds", allow_labels=True):
    X, y = apply_feature_args(df, [feature_arg], dataset_prefix=prefix, allow_labels=allow_labels)
    return X, y


@pytest.mark.parametrize(
    "feature_arg,warmup",
    [
        ("ema:20", 25),
        ("rsi:14", 20),
        ("volatility:20", 25),
        ("roc:5", 10),
        ("obv:20", 25),
        ("weekday", 0),
        ("month", 0),
        ("dollar_volume", 0),
        ("market_value", 0),
        ("shift:Close,Volume:1", 5),
    ],
)
def test_features_do_not_change_if_only_future_is_perturbed(feature_arg, warmup):
    """
    Core leakage guard:
    If we change values strictly AFTER a cutoff date, features at/before cutoff must be unchanged
    (for features that are causal/lookback-only).
    """
    df = make_ohlcv_df(n=80, seed=10)

    # For market_value needs SHROUT
    if feature_arg == "market_value":
        df = df.copy()
        df["SHROUT"] = 1000.0

    # Compute base
    X0, _ = _compute_feature(df, feature_arg, prefix="ds", allow_labels=True)

    # Perturb future only
    cutoff = df.index[warmup + 20]  # ensure we're past warmup and not at the end
    df2 = df.copy()
    future_mask = df2.index > cutoff
    df2.loc[future_mask, "Close"] = df2.loc[future_mask, "Close"] + 999.0
    df2.loc[future_mask, "Open"] = df2.loc[future_mask, "Open"] + 999.0
    df2.loc[future_mask, "High"] = df2.loc[future_mask, "High"] + 999.0
    df2.loc[future_mask, "Low"] = df2.loc[future_mask, "Low"] + 999.0

    X1, _ = _compute_feature(df2, feature_arg, prefix="ds", allow_labels=True)

    # Compare only <= cutoff, and only where both are not NaN
    cols = X0.columns.intersection(X1.columns)
    assert len(cols) > 0

    X0s = X0.loc[:cutoff, cols]
    X1s = X1.loc[:cutoff, cols]
    mask = (~X0s.isna()) & (~X1s.isna())

    # if feature is sparse NaNs at start, require at least some overlap
    assert mask.sum().sum() > 5
    assert np.allclose(X0s[mask].values, X1s[mask].values, atol=1e-10)


def test_fwd_ret_label_must_change_if_future_is_perturbed():
    """
    Labels are allowed to look into the future.
    If we perturb future opens, forward returns should change for earlier rows.
    """
    df = make_ohlcv_df(n=80, seed=11)

    # base label
    X0, y0 = _compute_feature(df, "fwd_ret:5", prefix="ds", allow_labels=True)
    # With allow_labels=True currently, fwd_ret ends up in X (depending on your implementation);
    # we still test the series existence and behavior.

    # perturb future opens only
    df2 = df.copy()
    cutoff = df.index[30]
    df2.loc[df2.index > cutoff, "Open"] = df2.loc[df2.index > cutoff, "Open"] + 1234.0

    X1, y1 = _compute_feature(df2, "fwd_ret:5", prefix="ds", allow_labels=True)

    # Wherever the fwd_ret lives, check it changes pre-cutoff (it should).
    # In your current architecture it likely lands in X unless separated into y.
    def pick_series(X, y):
        for frame in (y, X):
            if frame is not None and not frame.empty:
                # only column should be fwd_ret_5 or similar
                return frame.iloc[:, 0]
        raise AssertionError("No output series found for fwd_ret:5")

    s0 = pick_series(X0, y0)
    s1 = pick_series(X1, y1)

    # Pick some region where label is defined
    region = (s0.index >= df.index[5]) & (s0.index <= cutoff)
    # There should be at least one changed value
    assert (np.nan_to_num(s0[region].values) != np.nan_to_num(s1[region].values)).any()


# -----------------------
# 5) Pipeline-level "label separation" contract tests
# -----------------------
@pytest.mark.xfail(reason="Current apply_feature_args label routing likely inverted; this test enforces no-leak contract.")
def test_pipeline_puts_fwd_ret_in_y_not_X(tmp_path):
    """
    Contract we WANT:
      - With allow_labels=False, fwd_ret/fwd_ret_label must go to y, not X
    This prevents accidental leakage via labels being treated as features.
    """
    df = make_ohlcv_df(n=50, seed=12)
    p = tmp_path / "ds.csv"
    write_csv(df, p)

    specs = [
        DatasetSpec(
            name="ds",
            kind="daily_csv",
            source=p,
            date_col="Date",
            features=["ema:20", "fwd_ret:5"],
        )
    ]

    out = build_feature_table(specs, allow_labels=False, join="outer")
    X, y = out["X"], out["y"]

    assert not X.empty
    assert not y.empty
    assert not any("fwd_ret" in c for c in X.columns)
    assert any("fwd_ret" in c for c in y.columns)


@pytest.mark.xfail(reason="Negative shift periods cause future leakage; guard should reject periods < 0.")
def test_shift_negative_period_is_rejected():
    df = make_ohlcv_df(n=20, seed=13)
    # This should raise once you add a guard (periods must be >= 0).
    util_mod.shift("Close", -1, df)