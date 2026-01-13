import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .eval import aggregate_subject_logits
from .model import gradient_alignment_loss


def train_epoch_level(
        model,
        train_loader,
        val_ds,
        val_epoch_sids,
        fold_dir,
        epochs=100,
        lr=1e-3,
        lambda_grad=0.02,
        lambda_l1=0.0005,
        device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_auc = -1.0

    Xv, yv = val_ds.X, val_ds.y.numpy().ravel().astype(int)
    Sv = val_epoch_sids.astype(int)

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if len(batch) == 4:
                xb, yb, _, wb = batch
            else:
                xb, yb, _ = batch
                wb = None

            xb, yb = xb.to(device), yb.to(device)

            logits, theta, z = model(xb)
            loss_vec = bce(logits, yb)
            loss = (loss_vec * wb.to(device)).mean() if wb is not None else loss_vec.mean()

            total = loss + lambda_l1 * theta.abs().mean() + lambda_grad * gradient_alignment_loss(
                logits, theta, z
            )

            opt.zero_grad()
            total.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_v, _, _ = model(torch.as_tensor(Xv, dtype=torch.float32, device=device))
            logits_v = logits_v.cpu().numpy().ravel()

        subj_probs = aggregate_subject_logits(logits_v, Sv, method="trimmed", trim_p=0.1)
        subj_list = sorted(subj_probs.keys())
        y_true = np.array([int(np.round(yv[Sv == s].mean())) for s in subj_list])
        y_hat = np.array([subj_probs[s] for s in subj_list])

        from sklearn.metrics import roc_auc_score, average_precision_score

        try:
            auc = roc_auc_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan
            ap = average_precision_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            auc, ap = np.nan, np.nan

        if np.nan_to_num(auc, nan=-1.0) > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))

        print(f"[{os.path.basename(fold_dir)}] Epoch {ep:03d}  subjAUC={auc:.3f}  subjAP={ap:.3f}")
