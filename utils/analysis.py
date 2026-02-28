import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def check_distribution_shift(df, train_range, test_range):
    # Slice the data
    train_data = df.loc[train_range[0]:train_range[1]]
    test_data = df.loc[test_range[0]:test_range[1]]
    
    results = []
    
    for col in df.columns:
        # Run the KS 2-sample test
        stat, p_value = ks_2samp(train_data[col], test_data[col])
        
        results.append({
            "feature": col,
            "ks_stat": stat,
            "p_value": p_value,
            "is_drifted": p_value < 0.05
        })
    
    # Convert to DataFrame and sort by the most "drifted" features first
    results_df = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
    
    return results_df

def analyze_correlation_drift(X, y, train_bounds, test_bounds, features_to_track):
    """
    Analyzes Spearman correlation stability and sign flips between 
    train and test windows for a specific fold.
    """
    # 1. Subset the data
    X_train = X.loc[train_bounds[0]:train_bounds[1]]
    y_train = y.loc[train_bounds[0]:train_bounds[1]]
    X_test = X.loc[test_bounds[0]:test_bounds[1]]
    y_test = y.loc[test_bounds[0]:test_bounds[1]]
    
    # 2. Calculate Spearman Correlations
    corr_train = X_train.corrwith(y_train, method='spearman')
    corr_test = X_test.corrwith(y_test, method='spearman')
    
    # 3. Construct Drift DataFrame
    drift_df = pd.DataFrame({
        'corr_train': corr_train,
        'corr_test': corr_test
    })
    
    # 4. Calculate Stability Metrics
    drift_df['corr_diff'] = drift_df['corr_test'] - drift_df['corr_train']
    drift_df['abs_corr_diff'] = drift_df['corr_diff'].abs()
    drift_df['sign_flip'] = (np.sign(drift_df['corr_train']) != np.sign(drift_df['corr_test']))
    
    # Filter for the specific features of interest
    return drift_df.loc[features_to_track]