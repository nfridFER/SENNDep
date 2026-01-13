import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, f_classif

from .data import build_epoch_dataset
from .model import SENN
from .train import train_epoch_level
from .eval import evaluate_subject_level
from .explain import (
    collect_epoch_outputs,
    summarize_subject_concepts,
    label_concepts_by_correlation_epochs,
)
from .artifacts import save_fold_artifacts


def train_and_eval_fold(cfg, fold_idx: int, val_ids, id_to_file: dict, id_to_label: dict) -> dict:
    """
    One fold end-to-end:
      - build train/val epoch datasets (fit scaler/selector on train)
      - train SENN (select best by val subj-AUC)
      - evaluate on val
      - run explainability on val
      - save artifacts

    Returns: {"fold": int, "acc": float, "roc": float, "pr": float}
    """
    fold_dir = os.path.join(cfg.save_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    print(f"\n=== Fold {fold_idx} ===")

    val_files = [id_to_file[i] for i in val_ids]
    val_labels = [id_to_label[i] for i in val_ids]

    train_ids = [i for i in id_to_file.keys() if i not in val_ids]
    train_files = [id_to_file[i] for i in train_ids]
    train_labels = [id_to_label[i] for i in train_ids]

    selector = SelectKBest(
        f_classif,
        k=min(cfg.selector_k, max(5, int(0.8 * len(train_files)))),
    )

    train_ds, feat_names, scaler, selector, ch_ref, train_epoch_sids, train_sid_maps = build_epoch_dataset(
        train_files,
        train_labels,
        selector=selector,
        fit_selector=True,
        scaler=None,
        fit_scaler=True,
        epoch_sec=cfg.epoch_sec,
        overlap_sec=cfg.epoch_overlap_sec,
        subject_ids=train_ids,
        ch_names_ref=None,
    )

    val_ds, _, _, _, _, val_epoch_sids, val_sid_maps = build_epoch_dataset(
        val_files,
        val_labels,
        selector=selector,
        fit_selector=False,
        scaler=scaler,
        fit_scaler=False,
        epoch_sec=cfg.epoch_sec,
        overlap_sec=cfg.epoch_overlap_sec,
        subject_ids=val_ids,
        ch_names_ref=ch_ref,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    input_dim = train_ds.X.shape[1]
    model = SENN(input_dim=input_dim, concept_dim=cfg.concept_dim)

    # ---- TRAIN ----
    train_epoch_level(
        model=model,
        train_loader=train_loader,
        val_ds=val_ds,
        val_epoch_sids=val_epoch_sids,
        fold_dir=fold_dir,
        epochs=cfg.epochs,
        lr=cfg.lr,
        lambda_grad=cfg.lambda_grad,
        lambda_l1=cfg.lambda_l1,
    )

    # ---- EVAL ----
    acc, roc, pr, cm, subj_probs, subj_conf = evaluate_subject_level(
        model=model,
        ds=val_ds,
        epoch_sids=val_epoch_sids,
        model_path=os.path.join(fold_dir, "best_model.pt"),
        prefix=f"FOLD {fold_idx} VAL",
    )

    # ---- EXPLAINABILITY on VAL ----
    logits_v, th_v, z_v, sids_v = collect_epoch_outputs(model, val_ds, val_epoch_sids)

    subj_summary = summarize_subject_concepts(
        epoch_theta=th_v, epoch_z=z_v, epoch_sids=sids_v, k=5, method="trimmed", trim_p=0.1
    )

    concept_labels = label_concepts_by_correlation_epochs(
        ds=val_ds, epoch_z=z_v, feature_names=feat_names, topk=5
    )

    def _pretty_sid(i):
        return val_sid_maps["int_to_sid"].get(i, str(i))

    subj_summary_named = {_pretty_sid(i): v for i, v in subj_summary.items()}

    # Save subject-level probabilities (REC_ID, prob) + confidence
    int_to_sid = val_sid_maps["int_to_sid"]
    pairs = [(int_to_sid[i], float(p)) for i, p in sorted(subj_probs.items(), key=lambda x: x[0])]
    np.save(os.path.join(fold_dir, "val_subject_probs.npy"), np.array(pairs, dtype=object), allow_pickle=True)

    pairs_prob_conf = [(int_to_sid[i], float(subj_probs[i]), float(subj_conf[i])) for i in sorted(subj_probs.keys())]
    np.save(
        os.path.join(fold_dir, "val_subject_probs_with_confidence.npy"),
        np.array(pairs_prob_conf, dtype=object),
        allow_pickle=True,
    )

    subj_probs_named = {int_to_sid[i]: float(p) for i, p in subj_probs.items()}
    subj_conf_named = {int_to_sid[i]: float(c) for i, c in subj_conf.items()}

    save_fold_artifacts(
        fold_dir=fold_dir,
        scaler=scaler,
        selector=selector,
        feat_names=feat_names,
        ch_names_ref=ch_ref,
        train_sid_maps=train_sid_maps,
        val_sid_maps=val_sid_maps,
        concept_labels=concept_labels,
        subj_summary_named=subj_summary_named,
        subj_probs_named=subj_probs_named,
        subj_conf_named=subj_conf_named,
    )

    return {"fold": fold_idx, "acc": acc, "roc": roc, "pr": pr}
