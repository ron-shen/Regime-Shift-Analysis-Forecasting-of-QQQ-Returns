import pandas as pd
import numpy as np

def apply_drift_filter(X, y, oof_pos, cv_res, fold_dates, window=21, threshold=0.2):
    """
    Day-by-day drift filter using explicit fold indexing to prevent coefficient mismatch.
    """
    filtered_pos = oof_pos.copy()
    fold_summaries = []
    
    for fold_info in fold_dates:
        # Use the explicit ID to fetch coefficients
        f_idx = fold_info['fold_idx']
        coefs = np.abs(cv_res[f_idx]['coefficients'])
        total_coef = np.sum(coefs)
        
        # Pull dates
        t_start, t_end = fold_info['Train_Start'], fold_info['Train_End']
        v_start, v_end = fold_info['Val_Start'], fold_info['Val_End']
        
        # Anchor: Training correlation
        X_train = X.loc[t_start:t_end]
        y_train = y.loc[t_start:t_end]
        corr_train = X_train.corrwith(y_train).fillna(0)
        
        # Live Walk-Forward
        val_idx = oof_pos.loc[v_start:v_end].index
        fold_drift_values = []
        
        for current_date in val_idx:
            recent_X = X.loc[:current_date].iloc[-window:]
            recent_y = y.loc[:current_date].iloc[-window:]
            
            if len(recent_X) < window:
                continue
                
            corr_recent = recent_X.corrwith(recent_y).fillna(0)
            
            #Drift Calculation
            drift_features = []
            for feat in corr_train.index:
                r_t, r_r = corr_train[feat], corr_recent[feat]
                d = max(0, r_t - r_r) if r_t >= 0 else max(0, r_r - r_t)
                drift_features.append(d)
            
            weighted_malignant_drift = np.sum(np.array(drift_features) * coefs) / total_coef
            fold_drift_values.append(weighted_malignant_drift)
            
            if weighted_malignant_drift > threshold:
                filtered_pos.loc[current_date] = 0
        
        if fold_drift_values:
            fold_summaries.append({
                'Fold_ID': f_idx + 1,
                'Avg_Malignant_Drift': np.mean(fold_drift_values),
                'Max_Malignant_Drift': np.max(fold_drift_values),
                'Pct_Time_Stopped': (filtered_pos.loc[val_idx] == 0).mean() * 100
            })
                
    return filtered_pos, pd.DataFrame(fold_summaries)