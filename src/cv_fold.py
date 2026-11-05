"""Subject-wise outer CV with an inner validation holdout."""
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .artifacts import save_fold_artifacts
from .config import CONFIG as DEFAULT_CONFIG
from .data import build_epoch_dataset
from .eval import evaluate_subject_level
from .explain import collect_epoch_outputs, concept_feature_correlation_matrix
from .model import SENN
from .train import train_epoch_level
from .utils import set_seed


def run_training(data_folder, excel_path, config=None):
    """Run the notebook training cell; paths replace its local Windows paths."""
    CONFIG = {**DEFAULT_CONFIG, **(config or {})}
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["feature_cache_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["save_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
    set_seed(CONFIG["seed"])

    df = pd.read_excel(excel_path)
    df["REC_ID"] = df["REC_ID"].astype(str)
    df["label"] = df["DIAGNOSIS"].apply(
        lambda d: 0 if str(d).lower().strip() == "healthy" else 1
    )
    df["is_GA"] = df["REC_ID"].str.startswith("GA")

    # Map REC_ID -> file path + label exactly as in the old code.
    id_to_file = {}
    id_to_label = {}
    for fname in os.listdir(data_folder):
        if not fname.endswith(".set"):
            continue
        stem = os.path.splitext(fname)[0]
        cleaned = re.sub(
            r"^(EEG_|rawEEG_|EEGbefore_|EEGbefore)",
            "",
            stem,
            flags=re.IGNORECASE,
        )
        if cleaned in df["REC_ID"].values:
            id_to_file[cleaned] = os.path.join(data_folder, fname)
            id_to_label[cleaned] = int(
                df.loc[df["REC_ID"] == cleaned, "label"].values[0]
            )

    # Preserve the old file-discovery order; do not impose a new channel or
    # subject ordering.
    subject_ids = np.array(list(id_to_file.keys()), dtype=object)
    if len(subject_ids) == 0:
        raise RuntimeError(f"No .set files matched subject IDs in {excel_path}")
    subject_position = {
        str(subject_id): position
        for position, subject_id in enumerate(subject_ids)
    }

    def restore_folder_order(ids):
        """Restore the original os.listdir/id_to_file order within a split."""
        return np.array(
            sorted(ids, key=lambda subject_id: subject_position[str(subject_id)]),
            dtype=object,
        )

    def subject_stratum(subject_id):
        """Stratify by diagnosis and by the two healthy-control sources."""
        row = df.loc[df["REC_ID"] == str(subject_id)].iloc[0]
        if int(row["label"]) == 1:
            return "MDD"
        return "healthy_GA" if bool(row["is_GA"]) else "healthy_old"

    strata = np.array([subject_stratum(sid) for sid in subject_ids], dtype=object)
    outer_cv = StratifiedKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )

    def describe_split(name, ids):
        sub = df[df["REC_ID"].isin(list(ids))]
        print(
            f"  {name:5s}: n={len(ids):3d} | "
            f"MDD={(sub['label'] == 1).sum():2d} | "
            f"healthy={(sub['label'] == 0).sum():2d} | "
            f"GA healthy={((sub['label'] == 0) & sub['is_GA']).sum():2d}"
        )

    def save_subject_probabilities(fold_dir, split_name, probs, conf, sid_maps):
        int_to_sid = sid_maps["int_to_sid"]
        pairs = [
            (int_to_sid[i], float(probs[i]))
            for i in sorted(probs.keys())
        ]
        np.save(
            os.path.join(fold_dir, f"{split_name}_subject_probs.npy"),
            np.array(pairs, dtype=object),
            allow_pickle=True,
        )
        pairs_with_confidence = [
            (int_to_sid[i], float(probs[i]), float(conf[i]))
            for i in sorted(probs.keys())
        ]
        np.save(
            os.path.join(
                fold_dir, f"{split_name}_subject_probs_with_confidence.npy"
            ),
            np.array(pairs_with_confidence, dtype=object),
            allow_pickle=True,
        )

    fold_results = []

    # Each outer test fold is held out. Checkpoint selection uses only the
    # separate inner validation subjects. Scaler and F-statistic selector are
    # fit only on the inner training subjects.
    for fold_idx, (outer_train_idx, test_idx) in enumerate(
        outer_cv.split(subject_ids, strata)
    ):
        fold_dir = os.path.join(CONFIG["save_dir"], f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        set_seed(CONFIG["seed"] + fold_idx)

        outer_train_ids = subject_ids[outer_train_idx]
        outer_train_strata = strata[outer_train_idx]
        test_ids = subject_ids[test_idx]

        train_ids, val_ids = train_test_split(
            outer_train_ids,
            test_size=CONFIG["val_fraction"],
            random_state=CONFIG["seed"] + fold_idx,
            shuffle=True,
            stratify=outer_train_strata,
        )
        train_ids = restore_folder_order(train_ids)
        val_ids = restore_folder_order(val_ids)
        test_ids = restore_folder_order(test_ids)

        train_set, val_set, test_set = map(
            lambda values: set(map(str, values)),
            (train_ids, val_ids, test_ids),
        )
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert train_set | val_set | test_set == set(map(str, subject_ids))

        print(f"\n=== Outer fold {fold_idx} ===")
        describe_split("TRAIN", train_ids)
        describe_split("VAL", val_ids)
        describe_split("TEST", test_ids)

        split_manifest = {
            "fold": fold_idx,
            "outer_seed": CONFIG["seed"],
            "inner_seed": CONFIG["seed"] + fold_idx,
            "train_ids": [str(x) for x in train_ids],
            "val_ids": [str(x) for x in val_ids],
            "test_ids": [str(x) for x in test_ids],
        }
        with open(
            os.path.join(fold_dir, "split_subject_ids.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(split_manifest, f, indent=2)

        train_files = [id_to_file[str(i)] for i in train_ids]
        train_labels = [id_to_label[str(i)] for i in train_ids]
        val_files = [id_to_file[str(i)] for i in val_ids]
        val_labels = [id_to_label[str(i)] for i in val_ids]
        test_files = [id_to_file[str(i)] for i in test_ids]
        test_labels = [id_to_label[str(i)] for i in test_ids]

        # Select the configured feature count on inner training epochs only.
        selector = SelectKBest(f_classif, k=CONFIG["selector_k"])

        (
            train_ds,
            feat_names,
            scaler,
            selector,
            ch_ref,
            train_epoch_sids,
            train_sid_maps,
        ) = build_epoch_dataset(
            train_files,
            train_labels,
            selector=selector,
            fit_selector=True,
            scaler=None,
            fit_scaler=True,
            epoch_sec=CONFIG["epoch_sec"],
            overlap_sec=CONFIG["epoch_overlap_sec"],
            subject_ids=train_ids,
            ch_names_ref=None,
            cache_dir=CONFIG["feature_cache_dir"],
        )

        val_ds, _, _, _, _, val_epoch_sids, val_sid_maps = build_epoch_dataset(
            val_files,
            val_labels,
            selector=selector,
            fit_selector=False,
            scaler=scaler,
            fit_scaler=False,
            epoch_sec=CONFIG["epoch_sec"],
            overlap_sec=CONFIG["epoch_overlap_sec"],
            subject_ids=val_ids,
            ch_names_ref=ch_ref,
            cache_dir=CONFIG["feature_cache_dir"],
        )

        test_ds, _, _, _, _, test_epoch_sids, test_sid_maps = build_epoch_dataset(
            test_files,
            test_labels,
            selector=selector,
            fit_selector=False,
            scaler=scaler,
            fit_scaler=False,
            epoch_sec=CONFIG["epoch_sec"],
            overlap_sec=CONFIG["epoch_overlap_sec"],
            subject_ids=test_ids,
            ch_names_ref=ch_ref,
            cache_dir=CONFIG["feature_cache_dir"],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        input_dim = train_ds.X.shape[1]
        model = SENN(input_dim=input_dim, concept_dim=CONFIG["concept_dim"])

        # Only inner validation subjects are used for checkpoint selection.
        train_epoch_level(
            model,
            train_loader,
            val_ds,
            val_epoch_sids,
            fold_dir=fold_dir,
            epochs=CONFIG["epochs"],
            lr=CONFIG["lr"],
            lambda_grad=CONFIG["lambda_grad"],
            lambda_l1=CONFIG["lambda_l1"],
        )

        _, _, _, _, val_probs, val_conf = evaluate_subject_level(
            model,
            val_ds,
            val_epoch_sids,
            model_path=os.path.join(fold_dir, "best_model.pt"),
            prefix=f"FOLD {fold_idx} VALIDATION",
        )
        test_acc, test_roc, test_pr, test_cm, test_probs, test_conf = (
            evaluate_subject_level(
                model,
                test_ds,
                test_epoch_sids,
                model_path=os.path.join(fold_dir, "best_model.pt"),
                prefix=f"FOLD {fold_idx} TEST",
            )
        )

        save_subject_probabilities(
            fold_dir, "val", val_probs, val_conf, val_sid_maps
        )
        save_subject_probabilities(
            fold_dir, "test", test_probs, test_conf, test_sid_maps
        )

        # Complete signed associations; train and outer-test stay separate.
        # Rows follow concept indices; columns follow selected_features.npy.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, z_train, _ = collect_epoch_outputs(
            model, train_ds, train_epoch_sids, device=device
        )
        _, _, z_test, _ = collect_epoch_outputs(
            model, test_ds, test_epoch_sids, device=device
        )
        corr_train = concept_feature_correlation_matrix(train_ds, z_train)
        corr_test = concept_feature_correlation_matrix(test_ds, z_test)

        save_fold_artifacts(
            fold_dir=fold_dir,
            scaler=scaler,
            selector=selector,
            feat_names=feat_names,
            ch_names_ref=ch_ref,
            train_sid_maps=train_sid_maps,
            val_sid_maps=val_sid_maps,
            test_sid_maps=test_sid_maps,
            concept_feature_corr_train=corr_train,
            concept_feature_corr_test=corr_test,
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "acc": test_acc,
                "roc": test_roc,
                "pr": test_pr,
                "confusion_matrix": test_cm.tolist(),
            }
        )

    print("\n===== OUTER-TEST SUBJECT-LEVEL SUMMARY (threshold = 0.5) =====")
    for fr in fold_results:
        print(
            f"Fold {fr['fold']}: acc={fr['acc']:.3f}, "
            f"roc={fr['roc']:.3f}, pr={fr['pr']:.3f}"
        )
    print("Artifacts per fold at:", os.path.abspath(CONFIG["save_dir"]))
    return fold_results
