"""Predict a new EEGLAB recording with one saved baseline fold."""
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit

from .data import extract_patient_epoch_matrix
from .eval import prob_confidence
from .model import SENN
from .subject_explanations import (
    analyze_subject,
    feature_display_name,
    load_state_dict_compat,
    motif_group,
)


def predict_eeg_set(root_dir, set_path, *, fold=0, tau=0.70, batch_size=256,
                    device=None, output_json=None):
    """Use saved preprocessing and notebook-style additive explanations.

    Feature allocations use training correlations, so a new recording does not
    participate in its own concept signatures. All contributions are MDD logit
    units; a negative value opposes MDD, independent of the predicted label.
    """
    root = Path(root_dir)
    recording = Path(set_path)
    if not recording.is_file():
        raise FileNotFoundError(f"Missing EEG recording: {recording}")
    if fold < 0 or not np.isfinite(tau) or not 0 < tau <= 1 or batch_size < 1:
        raise ValueError("Fold must be nonnegative, tau in (0, 1], and batch size positive.")
    with (root / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    if fold >= int(config["n_folds"]):
        raise ValueError(f"Fold {fold} is outside the saved run's {config['n_folds']} folds")
    folder = root / f"fold_{fold}"
    required = ["best_model.pt", "scaler.joblib", "selector.joblib", "channels.npy",
                "selected_features.npy", "concept_feature_corr_train.npy"]
    missing = [folder / name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing inference artifacts:\n" + "\n".join(map(str, missing)))
    channels = np.load(folder / "channels.npy", allow_pickle=True).astype(str).tolist()
    feature_names = np.load(folder / "selected_features.npy", allow_pickle=True).astype(str)
    raw, actual_channels = extract_patient_epoch_matrix(
        str(recording), epoch_sec=config["epoch_sec"], overlap_sec=config["epoch_overlap_sec"],
    )
    if list(actual_channels) != channels:
        raise ValueError(f"EEG channel names/order differ from training: expected {channels}, got {actual_channels}")
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("EEG features must form a nonempty finite matrix")
    scaler = joblib.load(folder / "scaler.joblib")
    selector = joblib.load(folder / "selector.joblib")
    x = selector.transform(scaler.transform(raw)).astype(np.float32, copy=False)
    concept_dim = int(config["concept_dim"])
    if feature_names.ndim != 1 or x.shape[1] != len(feature_names):
        raise ValueError("Selected feature names do not match the transformed model inputs")
    correlations = np.load(folder / "concept_feature_corr_train.npy", allow_pickle=False)
    if correlations.shape != (concept_dim, len(feature_names)) or not np.isfinite(correlations).all():
        raise ValueError("Training concept correlations do not match the saved model/features")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SENN(input_dim=x.shape[1], concept_dim=concept_dim).to(device)
    model.load_state_dict(load_state_dict_compat(folder / "best_model.pt", device))
    model.eval()
    analysis = analyze_subject(model, x, batch_size=batch_size, device=device)
    values = analysis["concept_contribution"]
    if abs(analysis["subject_logit"] - analysis["logit_from_contributions"]) > 1e-5:
        raise RuntimeError("Concept contributions do not reconstruct the subject logit")

    concepts = []
    motifs = defaultdict(float)
    for concept, value in enumerate(values):
        corr = correlations[concept].astype(float)
        magnitude = np.abs(corr)
        if magnitude.max() <= 0:
            raise ValueError(f"Concept C{concept} has an empty training correlation signature")
        keep = np.flatnonzero(magnitude >= tau * magnitude.max())
        weights = magnitude[keep] / magnitude[keep].sum()
        features = []
        for index, weight in zip(keep, weights):
            feature = str(feature_names[index])
            group = motif_group(feature)
            contribution = float(value * weight)
            features.append({
                "feature_name": feature, "feature_display_name": feature_display_name(feature),
                "concept_feature_correlation": float(corr[index]),
                "correlation_arrow": "↑" if corr[index] >= 0 else "↓",
                "allocation_weight": float(weight), "allocated_contribution": contribution,
                "feature_motif": group,
            })
            motifs[group] += contribution
        concepts.append({
            "concept_id": concept, "concept_name": f"C{concept}",
            "contribution": float(value),
            "direction": "supporting" if value > 0 else "opposing" if value < 0 else "neutral",
            "features": sorted(features, key=lambda row: -abs(row["allocated_contribution"])),
        })
    if abs(sum(motifs.values()) - analysis["subject_logit"]) > 1e-5:
        raise RuntimeError("Motif contributions do not reconstruct the subject logit")
    probability = float(expit(analysis["subject_logit"]))
    result = {
        "set_path": str(recording.resolve()), "fold": int(fold),
        "p_mdd": probability, "predicted_label": int(probability >= 0.5),
        "pred_label": "MDD" if probability >= 0.5 else "Healthy",
        "confidence": prob_confidence(probability), "threshold": 0.5,
        "subject_logit": analysis["subject_logit"], "n_epochs": analysis["n_epochs"],
        "n_retained_epochs": analysis["n_retained_epochs"], "tau": float(tau),
        "correlation_source": "inner_training",
        "concepts": sorted(concepts, key=lambda row: -abs(row["contribution"])),
        "motifs": sorted([{"motif_name": name, "contribution": float(value)}
                          for name, value in motifs.items()], key=lambda row: -abs(row["contribution"])),
    }
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result


def print_prediction(result):
    print(f"\n=== INFERENCE: fold {result['fold']} ===")
    print("File:", result["set_path"])
    print(f"p(MDD)={result['p_mdd']:.4f}  prediction={result['pred_label']}  confidence={result['confidence']:.3f}")
    print(f"Retained epochs: {result['n_retained_epochs']}/{result['n_epochs']}")
    print("Top concepts (signed MDD logit contributions):")
    for row in result["concepts"][:5]:
        print(f"  {row['concept_name']}: {row['contribution']:+.4f}")
    print("Top motifs:")
    for row in result["motifs"][:5]:
        print(f"  {row['motif_name']}: {row['contribution']:+.4f}")
