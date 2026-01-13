import numpy as np
import torch
from scipy.stats import trim_mean


def feature_family(name: str) -> str:
    n = name.lower()
    if n.startswith("relpow_"):
        return "relpower"
    if "coh_alpha[" in n:
        return "coherence_alpha"
    if n.startswith("alpha_asym"):
        return "alpha_asymmetry"
    if "ratio_mean_theta_over_alpha" in n:
        return "ratio_theta_alpha"
    if "ratio_mean_theta_over_beta" in n:
        return "ratio_theta_beta"
    if any(k in n for k in ["sampen", "approx_entropy", "renyi_entropy", "shannon_h"]):
        return "entropy"
    if "higuchi_fd" in n:
        return "fractal_fd"
    if "c0_complexity" in n:
        return "c0_complexity"
    if any(m in n for m in ["mean[", "std[", "skew[", "kurtosis["]):
        return "stats"
    return "other"


@torch.no_grad()
def collect_epoch_outputs(model, ds, epoch_sids, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    X = torch.as_tensor(ds.X, dtype=torch.float32, device=device)
    logits, th, z = model(X)

    return (
        logits.squeeze(-1).cpu().numpy(),
        th.cpu().numpy(),
        z.cpu().numpy(),
        epoch_sids.astype(int),
    )


def summarize_subject_concepts(epoch_theta, epoch_z, epoch_sids, k=5, method="trimmed", trim_p=0.1):
    subj_summary = {}
    sids = np.unique(epoch_sids)

    per_epoch_contrib = epoch_theta * epoch_z
    epoch_topk = np.argsort(-np.abs(per_epoch_contrib), axis=1)[:, :k]

    for sid in sids:
        m = epoch_sids == sid
        Z = epoch_z[m]
        TH = epoch_theta[m]

        if method == "mean":
            aggZ, aggTH = Z.mean(axis=0), TH.mean(axis=0)
        elif method == "median":
            aggZ, aggTH = np.median(Z, axis=0), np.median(TH, axis=0)
        else:
            aggZ = np.apply_along_axis(lambda v: trim_mean(v, proportiontocut=trim_p / 2), 0, Z)
            aggTH = np.apply_along_axis(lambda v: trim_mean(v, proportiontocut=trim_p / 2), 0, TH)

        contrib = aggTH * aggZ
        top_idx = np.argsort(-np.abs(contrib))[:k]

        ep_topk = epoch_topk[m]
        stab = {int(ci): float((ep_topk == ci).any(axis=1).mean()) for ci in top_idx}

        subj_summary[int(sid)] = {
            "theta": aggTH,
            "z": aggZ,
            "contrib": contrib,
            "topk": [(int(i), float(contrib[i])) for i in top_idx],
            "stability": stab,
        }

    return subj_summary


def label_concepts_by_correlation_epochs(ds, epoch_z, feature_names, topk=5):
    X = ds.X
    C = epoch_z.shape[1]

    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    labels = []
    for c in range(C):
        zc = epoch_z[:, c]
        zc = (zc - zc.mean()) / (zc.std() + 1e-9)

        r = (Xn * zc[:, None]).mean(axis=0)
        idx = np.argsort(-np.abs(r))[:topk]

        rows = []
        for j in idx:
            sgn = 1.0 if r[j] >= 0 else -1.0
            rows.append((feature_names[j], sgn, float(abs(r[j]))))
        labels.append(rows)

    return labels
