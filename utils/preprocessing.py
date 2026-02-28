import pandas as pd
import pywt
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