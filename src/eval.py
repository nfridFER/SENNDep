import numpy as np
import torch
from collections import defaultdict
from scipy.stats import trim_mean


def aggregate_subject_logits(
        epoch_logits: np.ndarray,
        epoch_subject_ids: np.ndarray,
        method="trimmed",
        trim_p=0.1,
        confidence_weighting=False,
) -> dict:
    buckets = defaultdict(list)
    for lg, sid in zip(epoch_logits, epoch_subject_ids):
        buckets[int(sid)].append(float(lg))

    out = {}
    for sid, logs in buckets.items():
        logs = np.asarray(logs, dtype=np.float64)

        if confidence_weighting:
            w = np.clip(np.abs(logs), 1e-6, None)
            agg_log = (w * logs).sum() / w.sum()
        else:
            if method == "mean":
                agg_log = logs.mean()
            elif method == "median":
                agg_log = np.median(logs)
            else:
                agg_log = trim_mean(logs, proportiontocut=trim_p / 2)

        out[sid] = 1.0 / (1.0 + np.exp(-agg_log))
    return out


def prob_confidence(p: float) -> float:
    return float(min(1.0, max(0.0, 2.0 * abs(p - 0.5))))


def evaluate_subject_level(model, ds, epoch_sids, model_path, prefix="VAL"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    X = ds.X
    y = ds.y.numpy().ravel().astype(int)
    S = epoch_sids.astype(int)

    with torch.no_grad():
        logits, _, _ = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        logits = logits.cpu().numpy().ravel()

    subj_probs = aggregate_subject_logits(logits, S, method="trimmed", trim_p=0.1)
    subj_conf = {sid: prob_confidence(p) for sid, p in subj_probs.items()}

    subj_list = sorted(subj_probs.keys())
    y_true = np.array([int(np.round(y[S == s].mean())) for s in subj_list])
    y_hat = np.array([subj_probs[s] for s in subj_list])
    y_pred = (y_hat >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        confusion_matrix,
        classification_report,
    )

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan
    pr = average_precision_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan
    cm = confusion_matrix(y_true, y_pred)
    rpt = classification_report(y_true, y_pred, target_names=["Healthy", "MDD"], digits=3)

    print(f"\n=== {prefix} (subject-level) ===")
    print(f"Accuracy: {acc:.3f}  ROC-AUC: {roc:.3f}  PR-AUC: {pr:.3f}")
    print("Confusion matrix:\n", cm)
    print("Report:\n", rpt)

    return acc, roc, pr, cm, subj_probs, subj_conf
