"""Fixed-threshold outer-test reports and ROC/PR plots."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)



def summarize_cv_results(root_dir, excel_path, n_folds=5, decision_threshold=0.5):
    """Report fixed-threshold metrics and curves from saved outer-test predictions."""
    ROOT_DIR = os.fspath(root_dir)
    EXCEL_PATH = excel_path
    N_FOLDS = n_folds
    DECISION_THRESHOLD = decision_threshold

    REPORT_PATH = os.path.join(ROOT_DIR, "metrics_report.txt")
    ROC_FIG_PATH = os.path.join(ROOT_DIR, "ROC_curve.pdf")
    PR_FIG_PATH = os.path.join(ROOT_DIR, "PR_curve.pdf")

    os.makedirs(ROOT_DIR, exist_ok=True)


    # =========================
    # LABELS
    # =========================
    df = pd.read_excel(EXCEL_PATH)
    df["REC_ID"] = df["REC_ID"].astype(str)
    df["label"] = df["DIAGNOSIS"].apply(
        lambda d: 0 if str(d).lower().strip() == "healthy" else 1
    )

    if df["REC_ID"].duplicated().any():
        duplicated = sorted(df.loc[df["REC_ID"].duplicated(), "REC_ID"].unique())
        raise RuntimeError(f"Duplicate REC_ID values in metadata: {duplicated}")

    id_to_label = dict(zip(df["REC_ID"], df["label"]))


    # =========================
    # METRICS
    # =========================
    def compute_metrics(y_true, y_scores, threshold=DECISION_THRESHOLD):
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)
        y_pred = (y_scores >= threshold).astype(int)

        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_scores)
            average_precision = average_precision_score(y_true, y_scores)
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, y_scores
            )
            pr_auc = auc(recall_curve[::-1], precision_curve[::-1])
        else:
            roc_auc = np.nan
            pr_auc = np.nan
            average_precision = np.nan

        return {
            "acc": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "average_precision": average_precision,
        }


    def load_test_predictions(path):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise RuntimeError(f"Unexpected prediction array shape in {path}: {arr.shape}")

        rec_ids = [str(row[0]) for row in arr]
        missing_ids = sorted(set(rec_ids) - set(id_to_label))
        if missing_ids:
            raise RuntimeError(f"Prediction file contains unknown REC_ID values: {missing_ids}")
        if len(rec_ids) != len(set(rec_ids)):
            raise RuntimeError(f"Duplicate subjects within prediction file: {path}")

        y_scores = np.asarray([float(row[1]) for row in arr], dtype=float)
        if not np.all(np.isfinite(y_scores)):
            raise RuntimeError(f"Non-finite prediction scores in {path}")
        if np.any((y_scores < 0.0) | (y_scores > 1.0)):
            raise RuntimeError(f"Prediction scores outside [0, 1] in {path}")

        y_true = np.asarray([id_to_label[rec_id] for rec_id in rec_ids], dtype=int)
        return rec_ids, y_true, y_scores


    def mean_std(series):
        values = series.dropna()
        if len(values) == 0:
            return "nan"
        if len(values) == 1:
            return f"{values.iloc[0]:.3f}"
        return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


    # =========================
    # LOAD OUTER-TEST OUTPUTS
    # =========================
    fold_results = []
    fold_curves_roc = []
    fold_curves_pr = []
    all_test_ids = []
    all_test_y_true = []
    all_test_y_scores = []

    for fold_idx in range(N_FOLDS):
        fold_dir = os.path.join(ROOT_DIR, f"fold_{fold_idx}")
        test_path = os.path.join(fold_dir, "test_subject_probs.npy")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Missing outer-test predictions for fold {fold_idx}: {test_path}"
            )

        test_ids, y_test, test_scores = load_test_predictions(test_path)

        overlap = set(test_ids) & set(all_test_ids)
        if overlap:
            raise RuntimeError(
                f"Subjects occur in more than one outer-test fold: {sorted(overlap)}"
            )

        metrics = compute_metrics(y_test, test_scores)
        fold_results.append(
            {
                "fold": fold_idx,
                "n_subjects": len(y_test),
                "threshold": DECISION_THRESHOLD,
                **metrics,
            }
        )

        all_test_ids.extend(test_ids)
        all_test_y_true.append(y_test)
        all_test_y_scores.append(test_scores)

        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, test_scores)
            fold_curves_roc.append(
                {"fold": fold_idx, "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
            )

            precision_curve, recall_curve, _ = precision_recall_curve(
                y_test, test_scores
            )
            fold_curves_pr.append(
                {
                    "fold": fold_idx,
                    "precision": precision_curve,
                    "recall": recall_curve,
                    "pr_auc": auc(recall_curve[::-1], precision_curve[::-1]),
                    "ap": average_precision_score(y_test, test_scores),
                }
            )

    results_df = pd.DataFrame(fold_results)
    if len(results_df) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} complete folds, found {len(results_df)}")


    # =========================
    # POOLED OUT-OF-FOLD RESULTS
    # =========================
    pooled_y_true = np.concatenate(all_test_y_true)
    pooled_y_scores = np.concatenate(all_test_y_scores)
    pooled_metrics = compute_metrics(pooled_y_true, pooled_y_scores)
    prevalence = float((pooled_y_true == 1).mean())

    fpr_pool, tpr_pool, _ = roc_curve(pooled_y_true, pooled_y_scores)
    roc_auc_pool = auc(fpr_pool, tpr_pool)
    precision_pool, recall_pool, _ = precision_recall_curve(
        pooled_y_true, pooled_y_scores
    )
    pr_auc_pool = auc(recall_pool[::-1], precision_pool[::-1])
    ap_pool = average_precision_score(pooled_y_true, pooled_y_scores)


    # =========================
    # WRITE REPORT
    # =========================
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        def w(line=""):
            report.write(line + "\n")

        w("All predictions are from untouched outer-test subjects.")
        w("Checkpoints were selected using inner validation subjects only.")
        w(
            "Classification metrics use the prespecified decision threshold "
            f"of {DECISION_THRESHOLD:.1f}; no threshold was optimized."
        )
        w()
        w("=== Per-fold outer-test metrics ===")
        w(results_df.round(3).to_string(index=False))
        w()
        w("===== CROSS-VALIDATION SUMMARY (mean ± sample SD across folds) =====")
        for metric in [
            "acc",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "average_precision",
        ]:
            w(f"{metric:18s}: {mean_std(results_df[metric])}")
        w()
        w("===== POOLED OUT-OF-FOLD SUBJECTS =====")
        w(f"n_subjects        : {len(pooled_y_true)}")
        w(f"prevalence (y=1) : {prevalence:.3f}")
        for metric in [
            "acc",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "average_precision",
        ]:
            w(f"{metric:18s}: {pooled_metrics[metric]:.3f}")

    print("Wrote:", REPORT_PATH)


    # =========================
    # OUTER-TEST ROC CURVE
    # =========================
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    for curve in fold_curves_roc:
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            alpha=0.6,
            label=f"Fold {curve['fold']} (AUC={curve['auc']:.3f})",
        )

    ax.plot(
        fpr_pool,
        tpr_pool,
        linewidth=2.5,
        linestyle="--",
        label=f"Pooled out-of-fold (AUC={roc_auc_pool:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_FIG_PATH, facecolor="white")
    plt.show()
    print("Saved:", ROC_FIG_PATH)


    # =========================
    # OUTER-TEST PRECISION-RECALL CURVE
    # =========================
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")

    for curve in fold_curves_pr:
        ax.plot(
            curve["recall"],
            curve["precision"],
            alpha=0.6,
            label=(
                f"Fold {curve['fold']} "
                f"(PR-AUC={curve['pr_auc']:.3f}, AP={curve['ap']:.3f})"
            ),
        )

    ax.plot(
        recall_pool,
        precision_pool,
        linewidth=2.5,
        linestyle="--",
        label=(
            f"Pooled out-of-fold "
            f"(PR-AUC={pr_auc_pool:.3f}, AP={ap_pool:.3f})"
        ),
    )
    ax.hlines(
        prevalence,
        0,
        1,
        linestyles=":",
        linewidth=1.0,
        label=f"Baseline (prevalence={prevalence:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PR_FIG_PATH, facecolor="white")
    plt.show()
    print("Saved:", PR_FIG_PATH)
    print("All outputs in:", os.path.abspath(ROOT_DIR))
