import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_perf_stats(name: str, stats: dict):
    pf = stats["pf"]
    mdd = stats["mdd"]
    total_return = stats["total_return"]

    pf_str = f"{pf:.2f}" if np.isfinite(pf) else "∞"

    print(f"\n===== {name} =====")
    print(f"{'Profit Factor:':<18} {pf_str}")
    print(f"{'Max Drawdown:':<18} {mdd:.2%}")
    print(f"{'Total Return:':<18} {total_return:.2%}")

def plot_coeffs_subplots(cv_res, feature_names, top_n=30, ncols=3, figsize_per_ax=(6, 5)):
    """
    Plot fold coefficient barplots in a grid of subplots (max ncols per row).
    Coefficients sorted ascending. Optionally show only top_n by |coef|.
    """
    fold_ids = sorted(cv_res.keys())
    n = len(fold_ids)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for i, fold_id in enumerate(fold_ids):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        coefs = np.asarray(cv_res[fold_id]["coefficients"]).ravel()
        if len(coefs) != len(feature_names):
            raise ValueError(
                f"Fold {fold_id}: coef length ({len(coefs)}) != feature_names ({len(feature_names)}). "
                "Your preprocess feature names likely don't match model features."
            )

        df = pd.DataFrame({"feature": feature_names, "coef": coefs})

        # keep only top |coef| if requested
        if top_n is not None and top_n < len(df):
            df = df.loc[df["coef"].abs().nlargest(top_n).index]

        df = df.sort_values("coef", ascending=True)

        ax.barh(df["feature"], df["coef"])
        ax.axvline(0, linewidth=1)
        ax.set_title(f"Fold {fold_id + 1} (PF_val={cv_res[fold_id]['val_score']:.2f})")
        ax.tick_params(axis="y", labelsize=8)

    # turn off empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

def coeff_importance_summary(
    cv_res: dict,
    feature_names,
    top_x: int = 20,
    pre_folds=(0, 1, 2, 3),
    post_folds=(4, 5, 6, 7),
):
    feature_names = np.array(feature_names, dtype=str)

    fold_ids = sorted(cv_res.keys())
    coef_mat = np.vstack([np.asarray(cv_res[f]["coefficients"]).ravel() for f in fold_ids])

    if coef_mat.shape[1] != len(feature_names):
        raise ValueError(
            f"coef length ({coef_mat.shape[1]}) != feature_names ({len(feature_names)})."
        )

    imp_df = pd.DataFrame(np.abs(coef_mat).T, index=feature_names, columns=fold_ids)

    # top-x flags
    in_top = pd.DataFrame(False, index=feature_names, columns=fold_ids)
    for f in fold_ids:
        top_feats = imp_df[f].nlargest(top_x).index
        in_top.loc[top_feats, f] = True

    top_presence = pd.DataFrame({
        f"top{top_x}_count": in_top.sum(axis=1).astype(int),
        f"top{top_x}_frac": (in_top.sum(axis=1) / len(fold_ids)).astype(float),
    }).sort_values([f"top{top_x}_count", f"top{top_x}_frac"], ascending=False)

    avg_imp = pd.DataFrame({
        "avg_importance": imp_df.mean(axis=1),
        "median_importance": imp_df.median(axis=1),
        "std_importance": imp_df.std(axis=1),
    }).sort_values("avg_importance", ascending=False)

    pre_folds = [f for f in pre_folds if f in fold_ids]
    post_folds = [f for f in post_folds if f in fold_ids]

    pre_mean = imp_df[pre_folds].mean(axis=1) if pre_folds else pd.Series(np.nan, index=imp_df.index)
    post_mean = imp_df[post_folds].mean(axis=1) if post_folds else pd.Series(np.nan, index=imp_df.index)

    pre_top_frac = in_top[pre_folds].mean(axis=1) if pre_folds else pd.Series(np.nan, index=imp_df.index)
    post_top_frac = in_top[post_folds].mean(axis=1) if post_folds else pd.Series(np.nan, index=imp_df.index)

    eps = 1e-12
    regime_compare = pd.DataFrame({
        "pre_mean_imp": pre_mean,
        "post_mean_imp": post_mean,
        "delta_post_minus_pre": post_mean - pre_mean,
        "ratio_post_over_pre": (post_mean + eps) / (pre_mean + eps),
        f"pre_top{top_x}_frac": pre_top_frac,
        f"post_top{top_x}_frac": post_top_frac,
        f"delta_top{top_x}_frac": post_top_frac - pre_top_frac,
    }).sort_values("delta_post_minus_pre", ascending=False)

    return imp_df, in_top, top_presence, avg_imp, regime_compare