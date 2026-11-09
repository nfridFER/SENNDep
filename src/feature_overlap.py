"""Pairwise Jaccard overlap of fold-specific selected feature sets."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_feature_overlap(root_dir, n_folds=5, top_k=80):
    """Summarize pairwise Jaccard overlap of saved fold-specific feature sets."""
    JACCARD_ROOT_DIR = Path(root_dir)
    JACCARD_N_FOLDS = n_folds
    JACCARD_TOP_K = top_k

    fold_feature_sets = {}
    for fold_idx in range(JACCARD_N_FOLDS):
        feature_path = JACCARD_ROOT_DIR / f"fold_{fold_idx}" / "selected_features.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing selected features for fold {fold_idx}: {feature_path}")
        selected_names = np.load(feature_path, allow_pickle=True)
        if selected_names.ndim != 1 or len(selected_names) != JACCARD_TOP_K:
            raise ValueError(
                f"Expected {JACCARD_TOP_K} feature names in {feature_path}; "
                f"found shape {selected_names.shape}."
            )
        if not all(isinstance(name, str) and name for name in selected_names):
            raise ValueError(f"Expected nonempty feature-name strings in {feature_path}.")
        feature_set = set(selected_names.tolist())
        if len(feature_set) != JACCARD_TOP_K:
            raise ValueError(f"Duplicate feature names in {feature_path}.")
        fold_feature_sets[fold_idx] = feature_set

    jaccard_rows = []
    for fold_a, fold_b in combinations(range(JACCARD_N_FOLDS), 2):
        features_a, features_b = fold_feature_sets[fold_a], fold_feature_sets[fold_b]
        intersection_size = len(features_a & features_b)
        union_size = len(features_a | features_b)
        jaccard_rows.append({
            "fold_a": fold_a,
            "fold_b": fold_b,
            "intersection_size": intersection_size,
            "union_size": union_size,
            "jaccard": intersection_size / union_size,
        })

    feature_jaccard_df = pd.DataFrame(jaccard_rows)
    feature_jaccard_mean = feature_jaccard_df["jaccard"].mean()
    feature_jaccard_sd = feature_jaccard_df["jaccard"].std(ddof=1)
    feature_jaccard_sentence = (
        f"Pairwise Jaccard overlap between the five fold-specific top-{JACCARD_TOP_K} "
        f"feature sets was {feature_jaccard_mean:.3f} ± {feature_jaccard_sd:.3f} "
        f"(mean ± sample SD across {len(feature_jaccard_df)} fold pairs)."
    )

    feature_jaccard_csv_path = JACCARD_ROOT_DIR / "feature_selection_pairwise_jaccard.csv"
    feature_jaccard_report_path = JACCARD_ROOT_DIR / "feature_selection_jaccard_summary.txt"
    feature_jaccard_df.to_csv(feature_jaccard_csv_path, index=False)
    feature_jaccard_report_path.write_text(feature_jaccard_sentence + "\n", encoding="utf-8")

    print(feature_jaccard_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n" + feature_jaccard_sentence)
    print("\nSaved:", feature_jaccard_csv_path)
    print("Saved:", feature_jaccard_report_path)
