import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from utils.metrics import perf_stats_from_logrets
from utils.plotting import print_perf_stats
from utils.filter import apply_drift_filter

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

def evaluate_strategy_filter(X, y, oof_pos, cv_res, fold_dates, threshold, window=21, label="Set"):
    """
    Applies the malignant drift filter and prints performance stats 
    before and after filtering for a given set of folds.
    """
    # 1. Apply Filter
    filtered_pos_full, _ = apply_drift_filter(
        X, y, oof_pos, cv_res, fold_dates, window=window, threshold=threshold
    )
    
    # 2. Slice data to the specific fold range
    start_date = fold_dates[0]['Val_Start']
    end_date = fold_dates[-1]['Val_End']
    
    filtered_pos = filtered_pos_full.loc[start_date:end_date]
    original_oos_pos = oof_pos.loc[start_date:end_date]
    
    # 3. Helper to calculate returns and print stats
    def run_perf_analysis(pos_series, title_suffix):
        tmp_df = pd.concat([y, pos_series], axis=1)
        # Using log1p for stability as in your original code
        tmp_df['strategy_return'] = np.log1p(tmp_df['qqq_fwd_ret_1']) * tmp_df[0]
        
        strat_ret = tmp_df['strategy_return'].dropna()
        stats = perf_stats_from_logrets(strat_ret)
        
        print_perf_stats(f"[{label}] {title_suffix}", stats)
        
        return stats

    # 4. Execute Analysis
    stats_before = run_perf_analysis(original_oos_pos, "Before filtering")
    stats_after = run_perf_analysis(filtered_pos, "After filtering")
    
    return stats_before, stats_after

def analyze_covariate_shift(X, cv_results, folds_info, top_x=15):
    """
    Analyzes covariate shift for the most important features of each fold.
    
    Args:
        X (pd.DataFrame): The feature matrix with a DatetimeIndex.
        cv_results (list): List of dicts, each containing a 'coefficients' key.
        folds_info (list): List of dicts with 'train'/'val' date tuples and 'id'.
        top_x (int): Number of top features (by absolute coefficient) to analyze.
    """
    analysis_data = []

    for i, fold in enumerate(folds_info):
        # 1. Extract coefficients and identify Top X features
        # Note: We use abs() because both large positive and negative weights are 'important'
        coeffs = np.array(cv_results[i]['coefficients'])
        feat_importance = pd.Series(np.abs(coeffs), index=X.columns)
        top_features = feat_importance.nlargest(top_x).index.tolist()
        
        # 2. Slice the dataframe for the specific fold periods
        train_slice = X.loc[fold['train'][0] : fold['train'][1], top_features]
        val_slice = X.loc[fold['val'][0] : fold['val'][1], top_features]
        
        # 3. Perform KS-test on each of the top features
        p_vals = [ks_2samp(train_slice[feat], val_slice[feat]).pvalue for feat in top_features]

        # Apply Benjamini-Yekutieli correction for multiple comparisons
        # fdr_by is valid under arbitrary dependence between tests
        rejected, _, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_by')

        drifted_features = [feat for feat, sig in zip(top_features, rejected) if sig]
        
        # 4. Record the results
        analysis_data.append({
            "fold_id": fold['id'],
            "status": fold.get('status', 'unknown'),
            "top_x_limit": top_x,
            "drift_count": len(drifted_features),
            "drift_ratio": round(len(drifted_features) / top_x, 3),
            "drifted_names": ", ".join(drifted_features)
        })

    return pd.DataFrame(analysis_data)

def summarize_all_folds_correlation(X, y, cv_res_list, folds_info, top_x=15):
    """
    Runs the correlation drift analysis across all folds and returns a summary.
    """
    summary_results = []

    for i, fold in enumerate(folds_info):
        # Extract coefficients for this specific fold from cv_res
        coeffs = cv_res_list[i]['coefficients']
        
        # Run the drift analysis for this fold
        drift_df = analyze_correlation_drift_top(
            X, y, 
            train_bounds=fold['train'], 
            test_bounds=fold['val'], 
            coefficients=coeffs, 
            top_x=top_x
        )
        
        # Aggregate stats for the summary
        summary_results.append({
            "fold_id": fold['id'],
            "status": fold.get('status', 'unknown'),
            "avg_abs_diff": drift_df['abs_corr_diff'].mean(),
            "sign_flip_count": drift_df['sign_flip'].sum(),
            "sign_flip_ratio": drift_df['sign_flip'].mean(),
            "worst_flip_feat": drift_df.sort_values('abs_corr_diff', ascending=False).index[0] if not drift_df.empty else None
        })

    return pd.DataFrame(summary_results)


def analyze_correlation_drift_top(X, y, train_bounds, test_bounds, coefficients, top_x=15):
    """
    Analyzes Spearman correlation stability for the top features identified 
    by model importance (coefficients).
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_bounds: Tuple of (start, end) dates for training
        test_bounds: Tuple of (start, end) dates for testing
        coefficients: Array or Series of model coefficients for the fold
        top_x (int): Select the top X features based on absolute coefficient weight.
    """
    # 1. Identify Top X features from Model Coefficients
    feat_importance = pd.Series(np.abs(coefficients), index=X.columns)
    top_features = feat_importance.nlargest(top_x).index.tolist()
    
    # 2. Subset the data
    X_train = X.loc[train_bounds[0]:train_bounds[1], top_features]
    y_train = y.loc[train_bounds[0]:train_bounds[1]]
    X_test = X.loc[test_bounds[0]:test_bounds[1], top_features]
    y_test = y.loc[test_bounds[0]:test_bounds[1]]
    
    # 3. Calculate Spearman Correlations
    corr_train = X_train.corrwith(y_train, method='spearman')
    corr_test = X_test.corrwith(y_test, method='spearman')
    
    # 4. Construct Drift DataFrame
    drift_df = pd.DataFrame({
        'importance': feat_importance.loc[top_features],
        'corr_train': corr_train,
        'corr_test': corr_test
    })
    
    # 5. Calculate Stability Metrics
    drift_df['corr_diff'] = drift_df['corr_test'] - drift_df['corr_train']
    drift_df['abs_corr_diff'] = drift_df['corr_diff'].abs()
    
    # Sign flip logic
    drift_df['sign_flip'] = (np.sign(drift_df['corr_train']) != np.sign(drift_df['corr_test'])) & \
                            (drift_df['corr_train'] != 0) & (drift_df['corr_test'] != 0)
    
    return drift_df



def analyze_fold_concept_drift(X, y, cv_res_list, folds_info, top_x=15):
    """
    Computes 'Normal' and 'Adversarial' metrics and identifies specific features 
    causing performance collapse via Adversarial Drift (decay and sign flips).
    """
    summary_results = []

    for i, fold in enumerate(folds_info):
        # 1. Get coefficients and identify top X features
        coeffs = cv_res_list[i]['coefficients']
        feat_importance = pd.Series(np.abs(coeffs), index=X.columns)
        top_features = feat_importance.nlargest(top_x).index.tolist()
        
        # 2. Subset data and calculate correlations
        X_train = X.loc[fold['train'][0]:fold['train'][1], top_features]
        y_train = y.loc[fold['train'][0]:fold['train'][1]]
        X_test = X.loc[fold['val'][0]:fold['val'][1], top_features]
        y_test = y.loc[fold['val'][0]:fold['val'][1]]
        
        corr_train = X_train.corrwith(y_train, method='spearman')
        corr_test = X_test.corrwith(y_test, method='spearman')
        
        # 3. Create working DataFrame for this fold
        df = pd.DataFrame({
            'corr_train': corr_train,
            'corr_test': corr_test,
            'abs_diff': (corr_test - corr_train).abs()
        })
        
        # 4. Logic for "Adversarial" features (Signals that weakened or flipped)
        cond_pos_drift = (df['corr_train'] > 0) & (df['corr_test'] < df['corr_train'])
        cond_neg_drift = (df['corr_train'] < 0) & (df['corr_test'] > df['corr_train'])
        df['is_adversarial'] = cond_pos_drift | cond_neg_drift
        
        # Calculate Adversarial Drift Magnitude 
        # (Positive value = moving toward noise or the opposite direction)
        df['adversarial_drift'] = np.where(
            df['corr_train'] > 0,
            df['corr_train'] - df['corr_test'],
            df['corr_test'] - df['corr_train']
        )
        
        # Sign flips: signs are different AND neither is zero
        df['sign_flip'] = (np.sign(df['corr_train']) != np.sign(df['corr_test'])) & \
                          (df['corr_train'] != 0) & (df['corr_test'] != 0)

        # 5. Extract Feature Lists
        adversarial_features = df[df['is_adversarial']].index.tolist()
        sign_flip_features = df[df['sign_flip']].index.tolist()
        
        # 6. Append summary results
        summary_results.append({
            "fold_id": fold['id'],
            "status": fold.get('status', 'unknown'),
            
            # --- NORMAL RESULTS ---
            "all_avg_abs_diff": round(df['abs_diff'].mean(), 4),
            "all_sign_flips": df['sign_flip'].sum(),
            "all_flip_ratio": round(df['sign_flip'].sum() / top_x, 3),
            
            # --- ADVERSARIAL DRIFT RESULTS ---
            "adversarial_count": len(adversarial_features),
            "adversarial_avg_drift": round(df.loc[adversarial_features, 'adversarial_drift'].mean(), 4) if adversarial_features else 0,
            "adversarial_feature_ratio": round(len(adversarial_features) / top_x, 2),
            
            # --- FEATURE NAMES ---
            "adversarial_feature_names": ", ".join(adversarial_features),
            "sign_flip_feature_names": ", ".join(sign_flip_features)
        })

    return pd.DataFrame(summary_results)