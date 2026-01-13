import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
)


def best_threshold_f1(y_true, y_scores, n_steps=200):
    """Pick the threshold that maximizes F1 (simple grid search)."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5  # degenerate: only one class present

    ts = np.linspace(0.1, 0.9, n_steps)
    best_t, best_f1v = 0.5, -1.0

    for t in ts:
        y_pred = (y_scores >= t).astype(int)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v, best_t = f1v, t

    return float(best_t)


def compute_metrics(y_true, y_scores, threshold):
    """Compute classification metrics at a fixed threshold + threshold-free AUCs."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        roc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
    else:
        roc, pr_auc = np.nan, np.nan

    return acc, prec, rec, f1v, roc, pr_auc


def _mean_std(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "nan"
    return f"{s.mean():.3f} ± {s.std():.3f}"


def summarize_cv_results(cfg, root_dir: str = None, excel_path: str  = None) -> dict:
    """
    Summarize per-fold subject-level predictions saved by training pipeline.

    Expects each fold dir to contain:
      fold_k/val_subject_probs.npy  with rows (REC_ID, prob)

    Writes:
      metrics_report.txt
      ROC_curve.pdf
      PR_curve.pdf

    Returns a dict with paths + pooled metrics.
    """
    root_dir = root_dir or cfg.save_dir
    excel_path = excel_path or cfg.excel_path
    if excel_path is None:
        raise ValueError("excel_path is None. Provide cfg.excel_path or pass excel_path=...")

    n_folds = cfg.n_folds
    fixed_t = cfg.fixed_threshold

    report_path = os.path.join(root_dir, cfg.report_filename)
    roc_fig_path = os.path.join(root_dir, cfg.roc_fig_filename)
    pr_fig_path = os.path.join(root_dir, cfg.pr_fig_filename)

    # --- labels (single source of truth) ---
    df = pd.read_excel(excel_path)
    df["REC_ID"] = df["REC_ID"].astype(str)
    df["label"] = df["DIAGNOSIS"].apply(lambda d: 0 if str(d).lower().strip() == "healthy" else 1)
    id_to_label = dict(zip(df["REC_ID"], df["label"]))

    fixed_results = []
    opt_results = []

    fold_curves_roc = []
    fold_curves_pr = []

    all_y_true = []
    all_y_scores = []

    for k in range(n_folds):
        fold_dir = os.path.join(root_dir, f"fold_{k}")
        probs_path = os.path.join(fold_dir, "val_subject_probs.npy")
        if not os.path.exists(probs_path):
            print(f"[WARN] Missing {probs_path}, skipping fold {k}")
            continue

        arr = np.load(probs_path, allow_pickle=True)
        rec_ids = [str(r[0]) for r in arr]
        y_scores = np.array([float(r[1]) for r in arr], dtype=float)

        # map labels from excel
        missing = [rid for rid in rec_ids if rid not in id_to_label]
        if missing:
            raise KeyError(f"Some REC_IDs from {probs_path} not found in Excel labels. Example: {missing[:5]}")

        y_true = np.array([id_to_label[rid] for rid in rec_ids], dtype=int)

        all_y_true.append(y_true)
        all_y_scores.append(y_scores)

        # fixed threshold
        acc, prec, rec, f1v, roc, pr_auc = compute_metrics(y_true, y_scores, fixed_t)
        fixed_results.append(
            dict(
                fold=k,
                n_subjects=len(y_true),
                threshold=fixed_t,
                acc=acc,
                precision=prec,
                recall=rec,
                f1=f1v,
                roc_auc=roc,
                pr_auc=pr_auc,
            )
        )

        # per-fold optimal threshold by F1
        t_opt = best_threshold_f1(y_true, y_scores)
        acc2, prec2, rec2, f12, roc2, pr_auc2 = compute_metrics(y_true, y_scores, t_opt)
        opt_results.append(
            dict(
                fold=k,
                n_subjects=len(y_true),
                threshold=t_opt,
                acc=acc2,
                precision=prec2,
                recall=rec2,
                f1=f12,
                roc_auc=roc2,
                pr_auc=pr_auc2,
            )
        )

        # curves
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            fold_curves_roc.append({"fold": k, "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)})

            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
            fold_curves_pr.append(
                {"fold": k, "prec": prec_curve, "rec": rec_curve, "ap": average_precision_score(y_true, y_scores)}
            )
        else:
            print(f"[WARN] Fold {k}: only one class in y_true, skipping ROC/PR curves.")

    fixed_df = pd.DataFrame(fixed_results)
    opt_df = pd.DataFrame(opt_results)

    # pooled curves
    if len(all_y_true) > 0:
        y_true_pool = np.concatenate(all_y_true)
        y_scores_pool = np.concatenate(all_y_scores)

        fpr_pool, tpr_pool, _ = roc_curve(y_true_pool, y_scores_pool)
        roc_auc_pool = auc(fpr_pool, tpr_pool)

        prec_pool, rec_pool, _ = precision_recall_curve(y_true_pool, y_scores_pool)
        ap_pool = average_precision_score(y_true_pool, y_scores_pool)

        prevalence = (y_true_pool == 1).mean()
    else:
        fpr_pool = tpr_pool = prec_pool = rec_pool = None
        roc_auc_pool = ap_pool = None
        prevalence = None

    # write report
    os.makedirs(root_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        def w(line=""):
            f.write(line + "\n")

        w(f"ROOT_DIR: {os.path.abspath(root_dir)}")
        w(f"EXCEL:    {os.path.abspath(excel_path)}")
        w()

        w(f"=== Per-fold metrics (fixed threshold = {fixed_t:.3f}) ===")
        if len(fixed_df) > 0:
            w(fixed_df.round(3).to_string(index=False))
        else:
            w("No folds found.")
        w()

        w(f"===== SUMMARY (fixed threshold = {fixed_t:.3f}) =====")
        for m in ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            if m in fixed_df:
                w(f"{m:10s}: {_mean_std(fixed_df[m])}")
        w()

        w("=== Per-fold metrics (per-fold optimal threshold by F1) ===")
        if len(opt_df) > 0:
            w(opt_df.round(3).to_string(index=False))
        else:
            w("No folds found.")
        w()

        w("===== SUMMARY (per-fold optimal threshold) =====")
        for m in ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            if m in opt_df:
                w(f"{m:10s}: {_mean_std(opt_df[m])}")
        w()

        if len(opt_df) > 0:
            w("Per-fold optimal thresholds:")
            for _, row in opt_df.iterrows():
                w(f"Fold {int(row['fold'])}: t_opt = {row['threshold']:.3f}")
            w()

        if prevalence is not None:
            w(f"Pooled prevalence (y=1): {prevalence:.3f}")
        if roc_auc_pool is not None:
            w(f"Pooled ROC-AUC: {roc_auc_pool:.3f}")
        if ap_pool is not None:
            w(f"Pooled PR-AUC:  {ap_pool:.3f}")

    print("Wrote:", report_path)

    # plots
    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    for c in fold_curves_roc:
        ax.plot(c["fpr"], c["tpr"], alpha=0.6, label=f"Fold {c['fold']} (AUC={c['auc']:.3f})")

    if fpr_pool is not None:
        ax.plot(fpr_pool, tpr_pool, linewidth=2.5, linestyle="--", label=f"Pooled (AUC={roc_auc_pool:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_fig_path, facecolor="white")
    plt.close(fig)
    print("Saved:", roc_fig_path)

    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    for c in fold_curves_pr:
        ax.plot(c["rec"], c["prec"], alpha=0.6, label=f"Fold {c['fold']} (AP={c['ap']:.3f})")

    if prec_pool is not None:
        ax.plot(rec_pool, prec_pool, linewidth=2.5, linestyle="--", label=f"Pooled (AP={ap_pool:.3f})")

    if prevalence is not None:
        ax.hlines(prevalence, 0, 1, linestyles=":", linewidth=1.0, label=f"Baseline (prevalence={prevalence:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(pr_fig_path, facecolor="white")
    plt.close(fig)
    print("Saved:", pr_fig_path)

    print("All outputs in:", os.path.abspath(root_dir))

    return {
        "report_path": report_path,
        "roc_fig_path": roc_fig_path,
        "pr_fig_path": pr_fig_path,
        "pooled_roc_auc": roc_auc_pool,
        "pooled_ap": ap_pool,
        "prevalence": prevalence,
        "fixed_df": fixed_df,
        "opt_df": opt_df,
    }
