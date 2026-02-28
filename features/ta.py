import numpy as np
import talib as ta
import pandas as pd
import math

def STOK(window, df): 
    close = df['Close']
    low = df['Low']
    high = df['High']
    STOK = ((close - low.rolling(window).min()) / (high.rolling(window).max() - low.rolling(window).min())) * 100

    return STOK


def STOD(window, df):
    close = df['Close']
    low = df['Low']
    high = df['High']
    STOK = ((close - low.rolling(window).min()) / (high.rolling(window).max() - low.rolling(window).min())) * 100
    STOD = STOK.rolling(3).mean()

    return STOD


def volatility(window, df):
    ret = df['Close'].pct_change(1)
    trading_days = 252
    rolling_vol = ret.rolling(window=window).std() * np.sqrt(trading_days)

    return rolling_vol


def ivol(window, df):
    trading_days = 252
    ret = df['Close'].pct_change(1)
    rolling_var = ret.rolling(window=window).var()
    rolling_coi = 1 - (ret.rolling(window=window).corr(df['sprtrn'])) ** 2
    ivol = np.sqrt(rolling_var * rolling_coi) * np.sqrt(trading_days)

    return ivol


def beta(window, asset_col, benchmark_col, df):
    # 1) returns (no implicit fill)
    asset_ret = df[asset_col].pct_change(1, fill_method=None)
    bench_ret = df[benchmark_col].pct_change(1, fill_method=None)

    # 2) align on the same timestamps + keep only valid pairs
    a, b = asset_ret.align(bench_ret, join="inner")
    paired = pd.concat([a.rename("asset"), b.rename("bench")], axis=1).dropna()

    # 3) rolling stats on paired observations
    rolling_cov = paired["asset"].rolling(window=window, min_periods=window).cov(paired["bench"])
    rolling_var = paired["bench"].rolling(window=window, min_periods=window).var()

    # 4) beta, avoid inf when var == 0
    beta_obs = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)

    # 5) reindex to original df index
    out = beta_obs.reindex(df.index)
    out.name = f"beta_{window}_{asset_col}_vs_{benchmark_col}"
    return out


def beta_squared(window, df):
    b = beta(window, df)
    return b ** 2


def ema(window, df):
    return ta.EMA(df['Close'], window)


def macd(fast, slow, signal, df):
    macd_res, _, _ =  ta.MACD(df['Close'], fast, slow, signal)
    return macd_res


def roc(window, df):
    return ta.ROC(df['Close'], window)


def mom(window, df):
    return ta.MOM(df['Close'], window)


def rsi(window, df):
    return ta.RSI(df['Close'], window)


def weekday(df):
    return pd.Series(index=df.index, data=df.index.weekday)


def month(df):
    return pd.Series(index=df.index, data=df.index.month)


def market_value(df):
    return df['Close'] * df['SHROUT']


def dollar_volume(df):
    return df['Close'] * df['Volume']


def fwd_open_to_open_ret(period, df):
    entry = df["Open"].shift(-1)              # next day's open (B)
    exit_ = df["Open"].shift(-(1 + period))   # open after holding period (C if period=1)
    return (exit_ / entry) - 1


def fwd_ret_label(period, df):
    ret = fwd_open_to_open_ret(period, df)
    return ret.where(ret.isna(), (ret > 0).astype(int).replace({0: -1}))


def vol_change(period, df):
    return df['Volume'].pct_change(period)


def roc(period, df):
    return ta.ROC(df['Close'], period)


def norm_obv(lookback, df):
    """
    ratio(t) = sum_{k=0..lookback-1} sign(Close[t-k] - Close[t-k-1]) * Volume[t-k]
               / sum_{k=0..lookback-1} Volume[t-k]
    normOBV(t) = 100 * normal_cdf(0.6 * sqrt(lookback) * ratio(t)) - 50
    """
    close_col = "Close"
    volume_col = "Volume"
    out_col = "normOBV"

    def _normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if close_col not in df.columns or volume_col not in df.columns:
        raise KeyError(f"df must contain {close_col!r} and {volume_col!r}")

    close = df[close_col].to_numpy()
    vol = df[volume_col].to_numpy()
    n = len(df)

    out = [0.0] * n
    sqrt_lb = math.sqrt(float(lookback))

    # need t-k-1 to exist -> t starts at lookback
    for t in range(lookback, n):
        signed_vol = 0.0
        total_vol = 0.0

        for k in range(lookback):
            diff = close[t - k] - close[t - k - 1]
            v = float(vol[t - k])

            if diff > 0:
                signed_vol += v
            elif diff < 0:
                signed_vol -= v

            total_vol += v

        if total_vol > 0.0:
            ratio = signed_vol / total_vol
            out[t] = 100.0 * _normal_cdf(0.6 * sqrt_lb * ratio) - 50.0
        else:
            out[t] = 0.0

    s = pd.Series(out, index=df.index, name=out_col)
    return s


def diff(window, df):
    return df['Close'].diff(window)

def acceleration(window, df):
    dif = diff(window, df)
    acc = dif.diff(window)
    return acc

def intermarket_divergence(window, series1, series2):
    df = pd.concat([series1, series2], axis=1).dropna()
    # Rolling correlation between Yield changes and QQQ returns
    return df.iloc[:, 0].pct_change().rolling(window).corr(df.iloc[:, 1].pct_change())


