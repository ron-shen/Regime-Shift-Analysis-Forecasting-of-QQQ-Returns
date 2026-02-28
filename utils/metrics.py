import numpy as np
import pandas as pd

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