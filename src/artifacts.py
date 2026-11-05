"""Persist preprocessing, subject maps, and complete correlations."""
import json
import os

import joblib
import numpy as np


def save_fold_artifacts(
        fold_dir,
        scaler,
        selector,
        feat_names,
        ch_names_ref,
        train_sid_maps,
        val_sid_maps,
        test_sid_maps=None,
        concept_feature_corr_train=None,
        concept_feature_corr_test=None,
):
    """
    Persist fold artifacts needed for reproducibility and later interpretation:
      - preprocessing (scaler, selector, feature names, channel list)
      - subject ID maps (int <-> REC_ID)
      - complete signed concept-feature correlation matrices for train/test
        (rows: concepts; columns: selected_features.npy order)
    """
    os.makedirs(fold_dir, exist_ok=True)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
    if selector is not None:
        joblib.dump(selector, os.path.join(fold_dir, "selector.joblib"))

    if feat_names is not None:
        np.save(
            os.path.join(fold_dir, "selected_features.npy"),
            np.array(feat_names, dtype=object),
            allow_pickle=True,
        )

    if ch_names_ref is not None:
        np.save(
            os.path.join(fold_dir, "channels.npy"),
            np.array(ch_names_ref, dtype=object),
            allow_pickle=True,
        )

    def _dump_sid_maps(name, sid_maps):
        if sid_maps is None:
            return
        path = os.path.join(fold_dir, f"{name}_sid_maps.json")
        out = {
            "sid_to_int": {str(k): int(v) for k, v in sid_maps["sid_to_int"].items()},
            "int_to_sid": {str(k): str(v) for k, v in sid_maps["int_to_sid"].items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    _dump_sid_maps("train", train_sid_maps)
    _dump_sid_maps("val", val_sid_maps)
    _dump_sid_maps("test", test_sid_maps)

    for split_name, correlations in (
        ("train", concept_feature_corr_train),
        ("test", concept_feature_corr_test),
    ):
        if correlations is not None:
            np.save(
                os.path.join(fold_dir, f"concept_feature_corr_{split_name}.npy"),
                correlations,
                allow_pickle=False,
            )
