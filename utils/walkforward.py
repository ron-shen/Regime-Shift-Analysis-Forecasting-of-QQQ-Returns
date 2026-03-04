import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

def run_walk_forward_evaluation(folds, X, y, pg, preprocess, model, scorer=None):
    """
    Executes walk-forward optimization and evaluation for a given set of folds.
    """
    results = {}
    # Series to store predictions and positions
    all_pred = pd.Series(index=X.index, dtype=float)
    all_pos  = pd.Series(index=X.index, dtype=int)
    
    for fold_id, train_idx, val_idx in folds:
        best_train_pf = -np.inf
        best_params = None

        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold     = X.iloc[val_idx], y.iloc[val_idx]

        # 1. Hyperparameter Tuning on the training window of THIS fold
        for param in pg:
            pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
            pipeline.set_params(**param)
            
            pipeline.fit(X_train_fold, y_train_fold)
            # Use pf_scorer for in-sample tuning
            y_train_pred = pipeline.predict(X_train_fold)
            score_train = scorer(y_train_fold, y_train_pred)

            if score_train > best_train_pf:
                best_train_pf = score_train
                best_params = param

        # 2. Refit using best parameters and evaluate on the validation/test window
        pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
        pipeline.set_params(**best_params)
        pipeline.fit(X_train_fold, y_train_fold)

        y_pred = pipeline.predict(X_val_fold)
        score_val = scorer(y_val_fold, y_pred)

        # 3. Store Results
        val_index = X_val_fold.index
        all_pred.loc[val_index] = y_pred
        all_pos.loc[val_index]  = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))

        results[fold_id] = {
            "params": best_params,
            "train_score": float(best_train_pf),
            "val_score": float(score_val),
            "coefficients": pipeline.named_steps["model"].coef_,
        }

        print(
        f"Fold {fold_id+1} | "
        f"Train: {X_train_fold.index[0]} -> {X_train_fold.index[-1]} | "
        f"Val: {X_val_fold.index[0]} -> {X_val_fold.index[-1]} | "
        f"PF_train={best_train_pf:.4f} PF_val={score_val:.4f}"
    )

    return results, all_pred, all_pos