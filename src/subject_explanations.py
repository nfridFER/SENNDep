"""Export additive subject explanations with correlation-based feature allocation.

Uses the baseline's per-fold artifacts and raw feature cache, without retraining.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit

from .config import CONFIG
from .model import SENN


def display(frame):
    """Print a table to the console."""
    print(frame.to_string(index=False))


def load_state_dict_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def cache_filename(subject_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subject_id))
    return f"{safe}.npz"


def retained_trim_indices(logits, trim_fraction_total=0.10):
    logits = np.asarray(logits, dtype=float)
    order = np.argsort(logits)
    cut_each_tail = int((trim_fraction_total / 2.0) * len(logits))
    if cut_each_tail == 0:
        return order
    return order[cut_each_tail : len(logits) - cut_each_tail]


FEATURE_FAMILY = {
    "sampen": "Irregularity",
    "shannon_H": "Entropy",
    "approx_entropy": "Entropy",
    "renyi_entropy": "Entropy",
    "higuchi_fd": "Fractal",
    "c0_complexity": "Complexity",
    "mean": "Statistical",
    "std": "Statistical",
    "skew": "Statistical",
    "kurtosis": "Statistical",
    "relpow_delta": "DeltaPower",
    "relpow_theta": "ThetaPower",
    "relpow_alpha": "AlphaPower",
    "relpow_beta": "BetaPower",
    "relpow_gamma": "GammaPower",
    "ratio_mean_theta_over_alpha": "BandPowerRatio",
    "ratio_mean_theta_over_beta": "BandPowerRatio",
    "coh_alpha": "Connectivity",
    "alpha_asym": "Asymmetry",
}


CHANNEL_REGION = {
    "Fp1": "LeftFrontal", "F3": "LeftFrontal", "F7": "LeftFrontal",
    "Fp2": "RightFrontal", "F4": "RightFrontal", "F8": "RightFrontal",
    "Fz": "FrontalMid",
    "C3": "Central", "C4": "Central", "Cz": "Central",
    "P3": "Parietal", "P4": "Parietal", "Pz": "Parietal",
    "T3": "Temporal", "T4": "Temporal",
    "T5": "Posterior", "T6": "Posterior",
    "O1": "Posterior", "O2": "Posterior",
}


REGION_ORDER = {
    "LeftFrontal": 0, "RightFrontal": 1, "FrontalMid": 2,
    "Central": 3, "Parietal": 4, "Temporal": 5, "Posterior": 6,
}


def parse_feature_name(feature: str):
    feature = str(feature).strip()
    match = re.fullmatch(r"([^\[\]]+?)(?:\[([^\[\]]+)\])?", feature)
    if match is None:
        raise ValueError(f"Feature name has an unsupported format: {feature!r}")
    base_name = match.group(1).strip()
    channel_spec = match.group(2)
    channels = [] if channel_spec is None else [part.strip() for part in channel_spec.split("-")]
    if any(not channel for channel in channels):
        raise ValueError(f"Feature contains an empty channel name: {feature!r}")
    return base_name, channels


def channel_region(channel: str, feature: str) -> str:
    if channel not in CHANNEL_REGION:
        raise ValueError(f"Unknown channel {channel!r} in feature {feature!r}.")
    return CHANNEL_REGION[channel]


def ordered_region_pair(channels, feature: str):
    if len(channels) != 2:
        raise ValueError(
            f"Connectivity/asymmetry feature must contain exactly two channels: {feature!r}"
        )
    regions = [channel_region(channel, feature) for channel in channels]
    return sorted(regions, key=REGION_ORDER.__getitem__)


def motif_group(feature: str) -> str:
    base_name, channels = parse_feature_name(feature)
    if base_name not in FEATURE_FAMILY:
        raise ValueError(f"Unknown feature family for {feature!r} (base name {base_name!r}).")
    family = FEATURE_FAMILY[base_name]
    if family in {"Connectivity", "Asymmetry"}:
        region_a, region_b = ordered_region_pair(channels, feature)
        return f"{region_a}{region_b}_{family}"
    if not channels:
        return f"Global_{family}"
    if len(channels) != 1:
        raise ValueError(f"Single-channel feature must contain one channel: {feature!r}")
    return f"{channel_region(channels[0], feature)}_{family}"


def concept_to_motif_weights(corr_row, features, threshold, mapper):
    abs_r = np.abs(np.asarray(corr_row, dtype=float))
    if abs_r.size == 0 or abs_r.max() <= 0:
        return {}, 0
    keep = abs_r >= float(threshold) * abs_r.max()
    selected_weights = abs_r[keep]
    selected_features = np.asarray(features)[keep]
    denom = selected_weights.sum()
    grouped = defaultdict(float)
    for feature, weight in zip(selected_features, selected_weights):
        grouped[mapper(str(feature))] += float(weight / denom)
    return dict(grouped), int(keep.sum())


FEATURE_DISPLAY_NAME = {
    "sampen": "SampEn surrogate",
    "shannon_H": "Shannon entropy",
    "approx_entropy": "approximate entropy",
    "renyi_entropy": "Rényi entropy",
    "higuchi_fd": "Higuchi fractal dimension",
    "c0_complexity": "C0 complexity",
    "mean": "mean", "std": "standard deviation",
    "skew": "skewness", "kurtosis": "kurtosis",
    "relpow_delta": "relative delta power",
    "relpow_theta": "relative theta power",
    "relpow_alpha": "relative alpha power",
    "relpow_beta": "relative beta power",
    "relpow_gamma": "relative gamma power",
    "ratio_mean_theta_over_alpha": "mean theta/alpha power ratio",
    "ratio_mean_theta_over_beta": "mean theta/beta power ratio",
    "coh_alpha": "alpha coherence",
    "alpha_asym": "alpha asymmetry",
}


def feature_display_name(feature: str) -> str:
    base_name, channels = parse_feature_name(feature)
    label = FEATURE_DISPLAY_NAME.get(base_name, base_name)
    if len(channels) == 1:
        return f"{label} at {channels[0]}"
    if len(channels) == 2:
        return f"{label} between {channels[0]} and {channels[1]}"
    return label


def analyze_subject(model: nn.Module, x: np.ndarray, *, batch_size=256,
                    device="cpu", trim_fraction_total=0.10) -> dict:
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Subject inputs must be a nonempty, finite epoch-feature matrix.")
    if batch_size < 1 or not 0 <= trim_fraction_total < 1:
        raise ValueError("Batch size must be positive and trim fraction in [0, 1).")
    logits_all = []
    contribution_all = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            logits, theta, z = model(xb)
            logits_all.append(logits.cpu().numpy().ravel())
            contribution_all.append((theta * z).cpu().numpy())

    logits = np.concatenate(logits_all)
    contribution = np.vstack(contribution_all)
    keep = retained_trim_indices(logits, trim_fraction_total)
    subject_contribution = contribution[keep].astype(np.float64).mean(axis=0)
    subject_logit = float(logits[keep].astype(np.float64).mean())
    logit_from_contributions = float(subject_contribution.sum(dtype=np.float64))

    return {
        "n_epochs": int(len(logits)),
        "n_retained_epochs": int(len(keep)),
        "concept_contribution": subject_contribution,
        "subject_logit": subject_logit,
        "logit_from_contributions": logit_from_contributions,
    }


def load_prediction_file(path: Path, id_to_label: dict) -> pd.DataFrame:
    """Combine saved (REC_ID, probability) predictions with true labels."""
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] != 2 or not len(arr):
        raise ValueError(f"Unexpected prediction format in {path}: {arr.shape}")
    ids = arr[:, 0].astype(str)
    missing = sorted(set(ids) - set(id_to_label))
    if missing:
        raise ValueError(f"Prediction subjects missing from Excel metadata: {missing}")
    probabilities = arr[:, 1].astype(float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"Invalid prediction probabilities in {path}")
    return pd.DataFrame({
        "subject_id": ids,
        "true_label": [id_to_label[sid] for sid in ids],
        "probability_mdd": probabilities,
    })


def load_fold_test_inputs(fold_dir: Path, cache_dir: Path):
    """Load test subjects from cached features using fold_k preprocessing."""
    with (fold_dir / "split_subject_ids.json").open(encoding="utf-8") as f:
        split = json.load(f)
    ids = [str(sid) for sid in split["test_ids"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate test subject IDs in {fold_dir}")
    scaler = joblib.load(fold_dir / "scaler.joblib")
    selector = joblib.load(fold_dir / "selector.joblib")
    feature_names = np.load(fold_dir / "selected_features.npy", allow_pickle=True).astype(str)
    if feature_names.ndim != 1:
        raise ValueError(f"Expected a feature-name vector in {fold_dir}")
    subject_inputs = {}
    for subject_id in ids:
        cache_path = cache_dir / cache_filename(subject_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing cached epoch features for subject {subject_id}: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            x_raw = cached["X_e"].astype(np.float32, copy=False)
        if x_raw.ndim != 2 or not len(x_raw) or not np.isfinite(x_raw).all():
            raise ValueError(f"Invalid cached epoch-feature matrix in {cache_path}")
        x = selector.transform(scaler.transform(x_raw)).astype(np.float32, copy=False)
        subject_inputs[subject_id] = x
    return subject_inputs, feature_names


def export_subject_explanations(root_dir, excel_path, *, feature_cache_dir=None,
                                output_csv=None, tau=0.70, batch_size=256, device=None):
    """Export each outer-test subject once; return (explanations, checks).

    Model size and seed come from the run's config.json. Relative cache paths
    keep their training-working-directory meaning; pass an override if moved.
    """
    RUN_ROOT = Path(root_dir)
    if not RUN_ROOT.is_dir():
        raise FileNotFoundError(f"Missing requested run directory: {RUN_ROOT}")
    with (RUN_ROOT / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    if not np.isfinite(tau) or not 0 < tau <= 1 or batch_size < 1:
        raise ValueError("Tau must be in (0, 1] and batch size must be positive.")
    FEATURE_CACHE_DIR = Path(feature_cache_dir if feature_cache_dir is not None
                             else config.get("feature_cache_dir", CONFIG["feature_cache_dir"]))
    if not FEATURE_CACHE_DIR.is_dir():
        raise FileNotFoundError(f"Missing feature cache directory: {FEATURE_CACHE_DIR}; supply --feature_cache_dir if moved.")
    CONCEPT_DIM = int(config["concept_dim"])
    CONFIG_NAME = f"senn_c{CONCEPT_DIM}"
    TAU = float(tau)
    DECISION_THRESHOLD = 0.50
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    token = format(TAU, "g").replace(".", "p")
    OUTPUT_CSV = Path(output_csv) if output_csv is not None else (
        RUN_ROOT / "interpretability_analysis" / f"subject_explanations_c{CONCEPT_DIM}_tau{token}.csv"
    )
    folds = list(range(int(config["n_folds"])))
    if not folds:
        raise ValueError("The run must contain at least one fold.")

    def run_dir(fold):
        return RUN_ROOT / f"fold_{fold}"

    required = ["best_model.pt", "test_subject_probs.npy", "concept_feature_corr_test.npy",
                "split_subject_ids.json", "scaler.joblib", "selector.joblib", "selected_features.npy"]
    missing = [run_dir(fold) / name for fold in folds for name in required
               if not (run_dir(fold) / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(map(str, missing)))
    labels = pd.read_excel(excel_path)
    labels["REC_ID"] = labels["REC_ID"].astype(str)
    if labels["REC_ID"].duplicated().any():
        raise ValueError("Duplicate REC_ID values in Excel metadata")
    id_to_label = dict(zip(labels["REC_ID"], labels["DIAGNOSIS"].apply(
        lambda diagnosis: 0 if str(diagnosis).lower().strip() == "healthy" else 1)))

    concept_feature_rows = []
    motif_rows = []
    check_rows = []
    seen_subjects = set()

    for fold in folds:
        rd = run_dir(fold)
        subject_inputs, feature_names = load_fold_test_inputs(rd, FEATURE_CACHE_DIR)
        predictions = load_prediction_file(rd / "test_subject_probs.npy", id_to_label)
        if predictions["subject_id"].duplicated().any():
            raise RuntimeError(f"Duplicate subject predictions in fold {fold}")
        prediction_by_subject = predictions.set_index("subject_id")

        if set(subject_inputs) != set(prediction_by_subject.index):
            raise RuntimeError(f"Prediction/input subject mismatch in fold {fold}")
        overlap = seen_subjects.intersection(subject_inputs)
        if overlap:
            raise RuntimeError(f"Subjects appear in more than one test fold: {sorted(overlap)}")
        seen_subjects.update(subject_inputs)

        corr = np.load(
            rd / "concept_feature_corr_test.npy", allow_pickle=False
        ).astype(float)
        if not np.isfinite(corr).all():
            raise ValueError(f"Non-finite correlations in fold {fold}")
        if corr.shape != (CONCEPT_DIM, len(feature_names)):
            raise ValueError(
                f"Correlation/feature mismatch in fold {fold}: "
                f"{corr.shape} versus ({CONCEPT_DIM}, {len(feature_names)})"
            )

        input_dim = next(iter(subject_inputs.values())).shape[1]
        if input_dim != len(feature_names):
            raise ValueError(
                f"Model input/feature-name mismatch in fold {fold}: "
                f"{input_dim} versus {len(feature_names)}"
            )
        model = SENN(input_dim=input_dim, concept_dim=CONCEPT_DIM).to(DEVICE)
        model.load_state_dict(load_state_dict_compat(rd / "best_model.pt", DEVICE))
        model.eval()

        # Preserve the complete retained feature list separately for every concept.
        retained_by_concept = []
        all_motifs = set()
        for concept in range(CONCEPT_DIM):
            corr_row = np.asarray(corr[concept], dtype=float)
            abs_corr = np.abs(corr_row)
            if abs_corr.size == 0 or abs_corr.max() <= 0:
                raise RuntimeError(f"Concept C{concept} has an empty correlation signature in fold {fold}")
            keep = abs_corr >= TAU * abs_corr.max()
            retained_indices = np.flatnonzero(keep)
            retained_abs_corr = abs_corr[keep]
            weights = retained_abs_corr / retained_abs_corr.sum()

            records = []
            for feature_index, weight in zip(retained_indices, weights):
                feature = str(feature_names[feature_index])
                correlation = float(corr_row[feature_index])
                motif = motif_group(feature)
                records.append(
                    {
                        "feature_index": int(feature_index),
                        "feature_name": feature,
                        "feature_display_name": feature_display_name(feature),
                        "feature_motif": motif,
                        "concept_feature_correlation": correlation,
                        "absolute_correlation": float(abs(correlation)),
                        "correlation_direction": "positive" if correlation >= 0 else "negative",
                        "correlation_arrow": "↑" if correlation >= 0 else "↓",
                        "allocation_weight": float(weight),
                    }
                )
                all_motifs.add(motif)
            retained_by_concept.append(records)

        motif_names = sorted(all_motifs)
        motif_position = {name: j for j, name in enumerate(motif_names)}
        motif_weight_matrix = np.zeros((CONCEPT_DIM, len(motif_names)), dtype=float)
        for concept, records in enumerate(retained_by_concept):
            for record in records:
                motif_weight_matrix[concept, motif_position[record["feature_motif"]]] += record["allocation_weight"]
        if not np.allclose(motif_weight_matrix.sum(axis=1), 1.0, atol=1e-12, rtol=1e-10):
            raise RuntimeError(f"Concept-to-motif weights do not sum to one in fold {fold}")

        for subject_id, x in subject_inputs.items():
            explanation = analyze_subject(model, x, batch_size=batch_size, device=DEVICE)
            concept_values = explanation["concept_contribution"]
            meta = prediction_by_subject.loc[subject_id]
            probability = float(meta["probability_mdd"])
            reconstructed_probability = float(expit(explanation["subject_logit"]))
            predicted_label = int(probability >= DECISION_THRESHOLD)
            predicted_group = "Predicted MDD" if predicted_label == 1 else "Predicted healthy"

            common = {
                "subject_id": subject_id,
                "fold": int(fold),
                "model": CONFIG_NAME,
                "training_seed": int(config["seed"]) + int(fold),
                "tau": TAU,
                "true_label": int(meta["true_label"]),
                "probability_mdd": probability,
                "predicted_label": predicted_label,
                "predicted_group": predicted_group,
                "subject_logit": explanation["subject_logit"],
                "n_epochs": explanation["n_epochs"],
                "n_retained_epochs": explanation["n_retained_epochs"],
            }

            concept_order = np.argsort(-np.abs(concept_values))
            concept_rank = np.empty(CONCEPT_DIM, dtype=int)
            concept_rank[concept_order] = np.arange(1, CONCEPT_DIM + 1)
            concept_directions = np.where(
                concept_values > 0, "supporting", np.where(concept_values < 0, "opposing", "neutral")
            )
            concept_direction_rank = np.empty(CONCEPT_DIM, dtype=int)
            for direction in ("supporting", "opposing", "neutral"):
                indices = np.flatnonzero(concept_directions == direction)
                ordered = indices[np.argsort(-np.abs(concept_values[indices]))]
                concept_direction_rank[ordered] = np.arange(1, len(ordered) + 1)
            max_concept_allocation_error = 0.0

            for concept, concept_contribution in enumerate(concept_values):
                records = retained_by_concept[concept]
                allocated_values = np.asarray(
                    [concept_contribution * record["allocation_weight"] for record in records],
                    dtype=float,
                )
                max_concept_allocation_error = max(
                    max_concept_allocation_error,
                    abs(float(allocated_values.sum()) - float(concept_contribution)),
                )
                feature_order = np.argsort(-np.abs(allocated_values))
                feature_rank = np.empty(len(records), dtype=int)
                feature_rank[feature_order] = np.arange(1, len(records) + 1)

                for local_index, (record, allocated_contribution) in enumerate(zip(records, allocated_values)):
                    concept_feature_rows.append(
                        {
                            **common,
                            "record_type": "concept_feature",
                            "concept_id": int(concept),
                            "concept_name": f"C{concept}",
                            "concept_contribution": float(concept_contribution),
                            "concept_direction": str(concept_directions[concept]),
                            "concept_absolute_rank": int(concept_rank[concept]),
                            "concept_rank_within_direction": int(concept_direction_rank[concept]),
                            **record,
                            "allocated_contribution": float(allocated_contribution),
                            "feature_rank_within_concept": int(feature_rank[local_index]),
                        }
                    )

            motif_values = concept_values @ motif_weight_matrix
            motif_order = np.argsort(-np.abs(motif_values))
            motif_rank = np.empty(len(motif_names), dtype=int)
            motif_rank[motif_order] = np.arange(1, len(motif_names) + 1)
            motif_directions = np.where(
                motif_values > 0, "supporting", np.where(motif_values < 0, "opposing", "neutral")
            )
            motif_direction_rank = np.empty(len(motif_names), dtype=int)
            for direction in ("supporting", "opposing", "neutral"):
                indices = np.flatnonzero(motif_directions == direction)
                ordered = indices[np.argsort(-np.abs(motif_values[indices]))]
                motif_direction_rank[ordered] = np.arange(1, len(ordered) + 1)

            for motif_index, (motif, motif_contribution) in enumerate(zip(motif_names, motif_values)):
                motif_rows.append(
                    {
                        **common,
                        "record_type": "motif",
                        "motif_name": motif,
                        "motif_display_name": motif.replace("_", " "),
                        "motif_contribution": float(motif_contribution),
                        "motif_direction": str(motif_directions[motif_index]),
                        "motif_absolute_rank": int(motif_rank[motif_index]),
                        "motif_rank_within_direction": int(motif_direction_rank[motif_index]),
                    }
                )

            check_rows.append(
                {
                    "subject_id": subject_id,
                    "fold": int(fold),
                    "saved_probability": probability,
                    "reconstructed_probability": reconstructed_probability,
                    "model_logit": explanation["subject_logit"],
                    "concept_total": explanation["logit_from_contributions"],
                    "motif_total": float(motif_values.sum(dtype=np.float64)),
                    "max_concept_allocation_error": float(max_concept_allocation_error),
                }
            )

        print(
            f"Fold {fold}: {len(subject_inputs)} subjects, "
            f"{sum(len(records) for records in retained_by_concept)} concept-feature links, "
            f"{len(motif_names)} motif groups"
        )

    concept_features = pd.DataFrame(concept_feature_rows)
    motifs = pd.DataFrame(motif_rows)
    checks = pd.DataFrame(check_rows)
    print(f"Reconstructed explanations for {len(checks)} subjects.")

    check_columns = ["concept_total", "motif_total"]
    for column in check_columns:
        checks[f"{column}_error"] = np.abs(checks[column] - checks["model_logit"])
    checks["probability_error"] = np.abs(
        checks["saved_probability"] - checks["reconstructed_probability"]
    )

    max_logit_error = float(
        checks[[f"{column}_error" for column in check_columns]].to_numpy().max()
    )
    max_probability_error = float(checks["probability_error"].max())
    max_concept_allocation_error = float(checks["max_concept_allocation_error"].max())
    if max_logit_error > 1e-5:
        display(checks.sort_values("concept_total_error", ascending=False).head(20))
        raise RuntimeError(f"Logit decomposition check failed: max error={max_logit_error:.3e}")
    if max_probability_error > 1e-5:
        display(checks.sort_values("probability_error", ascending=False).head(20))
        raise RuntimeError(
            f"Saved/reconstructed probability check failed: max error={max_probability_error:.3e}"
        )

    if max_concept_allocation_error > 1e-10:
        display(checks.sort_values("max_concept_allocation_error", ascending=False).head(20))
        raise RuntimeError(
            "Within-concept feature allocations do not sum to their concept contribution: "
            f"max error={max_concept_allocation_error:.3e}"
        )

    concept_features["concept_name"] = "C" + concept_features["concept_id"].astype(str)
    concept_features["allocated_contribution"] = concept_features["allocated_contribution"].astype(float)
    concept_features["allocation_direction"] = np.select(
        [concept_features["allocated_contribution"] > 0, concept_features["allocated_contribution"] < 0],
        ["supporting", "opposing"],
        default="neutral",
    )

    # One CSV, ordered as requested: all concept-feature rows for a subject, then motif rows.
    concept_features["contribution"] = concept_features["allocated_contribution"]
    concept_features["direction"] = concept_features["allocation_direction"]
    concept_features["record_order"] = 0
    concept_features["primary_rank"] = concept_features["concept_absolute_rank"]
    concept_features["secondary_rank"] = concept_features["feature_rank_within_concept"]

    motifs["contribution"] = motifs["motif_contribution"]
    motifs["direction"] = motifs["motif_direction"]
    motifs["record_order"] = 1
    motifs["primary_rank"] = motifs["motif_absolute_rank"]
    motifs["secondary_rank"] = 0

    explanations = pd.concat([concept_features, motifs], ignore_index=True, sort=False)
    explanations = explanations.sort_values(
        ["fold", "subject_id", "record_order", "primary_rank", "secondary_rank"]
    ).reset_index(drop=True)
    explanations = explanations.drop(
        columns=["record_order", "primary_rank", "secondary_rank"]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(explanations):,} rows to {OUTPUT_CSV.resolve()}")
    print(f"Maximum logit decomposition error: {max_logit_error:.3e}")
    print(f"Maximum within-concept allocation error: {max_concept_allocation_error:.3e}")
    print(f"Maximum saved/reconstructed probability error: {max_probability_error:.3e}")
    display(explanations.head(20))
    return explanations, checks


def show_subject(explanations: pd.DataFrame, subject_id: str,
                 n_concepts: int = 3, n_motifs_each: int = 5):
    """Print a subject's leading concepts, retained features, and motifs."""
    concept_features = explanations[explanations["record_type"] == "concept_feature"]
    motifs = explanations[explanations["record_type"] == "motif"]
    subject_cf = concept_features[concept_features["subject_id"] == str(subject_id)]
    subject_motifs = motifs[motifs["subject_id"] == str(subject_id)]
    if subject_cf.empty:
        raise KeyError(f"Unknown subject_id: {subject_id}")

    dominant = (
        subject_cf[["concept_id", "concept_name", "concept_contribution", "concept_absolute_rank"]]
        .drop_duplicates()
        .sort_values("concept_absolute_rank")
        .head(n_concepts)
    )
    for row in dominant.itertuples(index=False):
        print(f"\n{row.concept_name}: {row.concept_contribution:+.3f}")
        display(
            subject_cf[subject_cf["concept_id"] == row.concept_id]
            .sort_values("feature_rank_within_concept")[
                ["correlation_arrow", "feature_display_name", "allocated_contribution"]
            ]
        )

    print("\nDominant motifs")
    for direction in ("supporting", "opposing"):
        print(f"  {direction}:")
        display(
            subject_motifs[subject_motifs["motif_direction"] == direction]
            .sort_values("motif_rank_within_direction")
            .head(n_motifs_each)[["motif_display_name", "motif_contribution"]]
        )


