#!/usr/bin/env python3
"""
feature_engineering_cli.py

A CLI tool to apply multiple feature-engineering routines (beta, SMA, shift) on an OHLCV dataset and save the augmented data as CSV.
"""

import argparse
import pandas as pd
from features.ta import *
from utils.utility import shift



# --- Feature map ---
FEATURE_FUNCTIONS = {
    'beta': beta,
    'beta_squared': beta_squared,
    'shift': shift,
    'STOK': STOK,
    'STOD': STOD,
    'volatility': volatility,
    'ivol': ivol,
    'ema': ema,
    'macd': macd,
    'roc': roc,
    'mom': mom,
    'rsi': rsi,
    'weekday': weekday,
    'month': month,
    'market_value': market_value,
    'dollar_volume': dollar_volume,
    'fwd_ret': fwd_open_to_open_ret,
    'fwd_ret_label': fwd_ret_label,
    'vol_change': vol_change,
    'obv': norm_obv,
    'diff': diff,
    'acc': acceleration,
    'intermarket_div': intermarket_divergence,
}


def parse_feature_arg(feature_arg):
    """
    Parse a feature argument string of the form:
      name:param1:param2...

    Returns:
        tuple: (feature_name, [param1, param2, ...])
    """
    parts = feature_arg.split(':')
    name = parts[0]
    params = parts[1:]
    return name, params


def main():
    parser = argparse.ArgumentParser(
        description="Apply feature engineering to a CSV dataset"
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input CSV file (date-indexed, with OHLCV and any additional columns)')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save the output CSV with engineered features')
    parser.add_argument('-f', '--features', nargs='+', required=True,
                        help=(
                            "Features to apply. Examples:\n"
                            "  beta:20                -> rolling beta with window=20\n"
                            "  sma:50:Close           -> 50-period SMA of 'Close'\n"
                            "  shift:Close,Volume:1   -> shift 'Close' & 'Volume' by 1 (overwrite)\n"
                            "  STOK:14                -> %K Stochastic Oscillator (14)\n"
                            "  STOD:14                -> %D Stochastic Oscillator (14)\n"
                            "  volatility:20          -> annualized vol over 20 days\n"
                            "  weekday                -> weekday index (no params)\n"
                            "  fwd_ret:5       -> 5-day forward return\n"
                        ))
    
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.input, index_col='Date', parse_dates=True)

    # Apply each requested feature
    for feature_arg in args.features:
        name, params = parse_feature_arg(feature_arg)
        if name not in FEATURE_FUNCTIONS:
            raise ValueError(
                f"Unknown feature '{name}'. Available: {list(FEATURE_FUNCTIONS.keys())}"
            )
        func = FEATURE_FUNCTIONS[name]

        # Convert numeric parameters where appropriate
        processed_params = []
        for p in params:
            if p.isdigit():
                processed_params.append(int(p))
            else:
                try:
                    processed_params.append(float(p))
                except ValueError:
                    processed_params.append(p)

        # Run the feature function
        result = func(*processed_params, df)

        # Assign results back to df
        if isinstance(result, pd.Series):
            if processed_params:
                col_name = f"{name}_{processed_params[0]}"
            else:
                col_name = name
            df[col_name] = result
            print(f"Applied feature '{name}' -> column '{col_name}'")
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                df[col] = result[col]
            print(f"Applied feature '{name}' -> replaced/added columns {list(result.columns)}")
        else:
            raise ValueError(
                f"Feature function '{name}' must return a Series or DataFrame, got {type(result)}"
            )

    # Save augmented dataset
    df.to_csv(args.output)
    print(f"Saved engineered dataset to: {args.output}")


if __name__ == '__main__':
    main()
