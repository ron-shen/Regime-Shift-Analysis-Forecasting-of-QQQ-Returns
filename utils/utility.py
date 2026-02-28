import pandas as pd
import pywt
from scipy.special import ndtr
import numpy as np

def wavelet_smoother(x_train, scale=0.1):
        wavelet = "db6"
        x_train = pd.DataFrame(x_train)
        df_wavelets = x_train.copy()
        data_len = len(df_wavelets)
        
        for i in x_train.columns:
            signal = x_train[i]
            coefficients = pywt.wavedec(signal, wavelet, mode='smooth')
            coefficients[1:] = [pywt.threshold(i, value=scale*signal.max(), mode='soft') for i in coefficients[1:]]
            reconstructed_signal = pywt.waverec(coefficients, wavelet, mode='smooth')
            #if the input is odd, the reconstructed signal is 1 unit longer than the origional signal
            df_wavelets[i] = reconstructed_signal[:data_len]
        
        df_wavelets = df_wavelets.fillna(0)
        
        return df_wavelets


def shift(columns, periods, df):
    """
    Shift one or more columns by a given number of periods, replacing the original columns.

    Args:
        columns (str): Comma-separated column names to shift.
        periods (int): Number of periods to shift.
        df (pd.DataFrame): DataFrame containing the columns.

    Returns:
        pd.DataFrame: DataFrame of shifted columns (same names as inputs).
    """
    periods = int(periods)
    cols = [col.strip() for col in columns.split(',')]
    shifted = {}
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        shifted[col] = df[col].shift(periods)
    # Return a DataFrame where each column has the original name
    return pd.DataFrame(shifted, index=df.index)


def pf_scorer(y_true, y_pred, threshold=0):
    """
    Calculates Profit Factor considering a conviction threshold.
    If |y_pred| <= threshold, the position is 0 (Flat).
    """
    # Create an array of zeros with the same shape as y_pred
    sign = np.zeros_like(y_pred)
    
    # Only go Long (1) if prediction > threshold
    sign[y_pred > threshold] = 1
    
    # Only go Short (-1) if prediction < -threshold
    sign[y_pred < -threshold] = -1
    
    # Calculate strategy returns
    strategy_ret = y_true * sign
    
    # Filter for winning and losing trades
    profit = strategy_ret[strategy_ret > 0].sum()
    loss = strategy_ret[strategy_ret < 0].sum()
    
    return profit / abs(loss)

def pf_scorer_dynamic(y_true, y_pred, threshold_values):
    """
    y_true: Series of actual returns
    y_pred: Array of predicted returns
    threshold_values: Series or Array of thresholds aligned with y_true
    """
    # Force y_pred to be a series for easy alignment if it isn't
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    
    # Determine positions based on the threshold for each specific day
    sign = np.zeros_like(y_pred)
    sign[y_pred_series > threshold_values] = 1
    sign[y_pred_series < -threshold_values] = -1
    
    strategy_ret = y_true * sign
    
    profit = strategy_ret[strategy_ret > 0].sum()
    loss = strategy_ret[strategy_ret < 0].sum()
    
    if abs(loss) < 1e-10:
        return profit if profit > 0 else 1.0
        
    return profit / abs(loss)

class SlidingWindowCV:
    """
    Fixed-size sliding-window CV:
      - train window: [t, t+train_size-1]
      - test  window: [t+train_size, t+train_size+test_size-1]
      - step controls how far we slide each fold (default = test_size)
    """
    def __init__(self, train_size, test_size, step=None, start=0, end=None):
        self.train_size = int(train_size)
        self.test_size  = int(test_size)
        self.step       = int(step) if step is not None else int(test_size)
        self.start      = int(start)
        self.end        = None if end is None else int(end)

    def split(self, X, y=None, groups=None):
        n = len(X)
        end = n if self.end is None else min(self.end, n)
        t = self.start
        while True:
            tr_start = t
            tr_end   = t + self.train_size            # exclusive
            te_start = tr_end
            te_end   = te_start + self.test_size      # exclusive
            if te_end > end:
                break
            train_idx = np.arange(tr_start, tr_end)
            test_idx  = np.arange(te_start, te_end)
            yield train_idx, test_idx
            t += self.step

    def get_n_splits(self, X=None, y=None, groups=None):
        # Optional; sklearn can work without this.
        n = len(X) if X is not None else 0
        end = n if self.end is None else min(self.end, n)
        t = self.start
        k = 0
        while True:
            if t + self.train_size + self.test_size > end:
                break
            k += 1
            t += self.step
        return k
    

def perf_stats_from_logrets(log_rets: pd.Series,
                            start=None,
                            end=None) -> dict:
    if not isinstance(log_rets, pd.Series):
        raise TypeError("log_rets must be a pandas Series.")

    s = log_rets.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    if start is not None or end is not None:
        s = s.loc[start:end]

    s = s.dropna()
    if s.empty:
        return {"pf": np.nan, "mdd": np.nan, "total_return": np.nan}

    # Profit Factor
    pos_sum = s[s > 0].sum()
    neg_sum = s[s < 0].sum()  # negative
    if neg_sum == 0:
        pf = np.inf if pos_sum > 0 else np.nan
    else:
        pf = pos_sum / (-neg_sum)

    # Convert log returns -> equity curve in simple terms
    equity = np.exp(s.cumsum())  # starts at 1.0 implicitly
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = drawdown.min()  # most negative

    # Total return over period (simple)
    total_return = equity.iloc[-1] - 1.0

    return {"pf": float(pf), "mdd": float(mdd), "total_return": float(total_return)}

def print_perf_stats(name: str, stats: dict):
    pf = stats["pf"]
    mdd = stats["mdd"]
    total_return = stats["total_return"]

    pf_str = f"{pf:.2f}" if np.isfinite(pf) else "∞"

    print(f"\n===== {name} =====")
    print(f"{'Profit Factor:':<18} {pf_str}")
    print(f"{'Max Drawdown:':<18} {mdd:.2%}")
    print(f"{'Total Return:':<18} {total_return:.2%}")

