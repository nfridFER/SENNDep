"""Complete signed train/test concept-feature correlation exports."""
import numpy as np
import torch


@torch.no_grad()
def collect_epoch_outputs(model, ds, epoch_sids, device=None):
    """
    Collect epoch-wise SENN outputs for a dataset split.

    Returns:
      logits: [N_epochs]
      theta:  [N_epochs, C]
      z:      [N_epochs, C]
      sids:   [N_epochs] (int)
    """
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


def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def concept_feature_correlation_matrix(ds, epoch_z):
    """
    Complete signed Pearson-correlation matrix [n_concepts, n_features].

    Nothing is thresholded here. Human-readable top-k labels and later motif
    thresholds can be derived from this lossless artifact.
    """
    # Preserve the notebook baseline normalization exactly:
    # ds.X remains a torch tensor (including torch.std's convention), while
    # each concept activation is standardized with NumPy as in the baseline.
    X = ds.X
    Xn = (X - X.mean(axis=0, keepdims=True)) / (
        X.std(axis=0, keepdims=True) + 1e-9
    )

    rows = []
    for c in range(epoch_z.shape[1]):
        zc = epoch_z[:, c]
        zc = (zc - zc.mean()) / (zc.std() + 1e-9)
        r = (Xn * zc[:, None]).mean(axis=0)
        rows.append(_as_numpy(r))
    return np.asarray(rows, dtype=np.float32)
