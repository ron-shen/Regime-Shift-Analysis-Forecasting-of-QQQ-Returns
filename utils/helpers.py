import numpy as np
import pandas as pd

def round_sigfig_df(df: pd.DataFrame, sig=3) -> pd.DataFrame:   
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(lambda x: np.nan if pd.isna(x) else float(f"{x:.{sig}g}"))
    return out


def get_preprocessed_feature_names(preprocess, X):
    """Fit preprocess once and pull expanded feature names."""
    try:
        preprocess.fit(X)
        names =  np.array(preprocess.get_feature_names_out())
        names = np.array([n.split("__", 1)[-1] for n in names], dtype=str)
        return names
    except Exception:
        return np.array(getattr(X, "columns", [f"f{i}" for i in range(X.shape[1])]))