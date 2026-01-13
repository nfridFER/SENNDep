from __future__ import annotations
from dataclasses import dataclass
import os
import json


@dataclass(frozen=True)
class Config:
    seed: int = 42
    epoch_sec: float = 2.0
    epoch_overlap_sec: float = 0.5
    selector_k: int = 80
    lambda_grad: float = 0.02
    lambda_l1: float = 0.0005
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 100
    concept_dim: int = 50
    save_dir: str = "./senn_mdd_testrun"
    n_folds: int = 5

    # --- reporting  ---
    excel_path: str = None
    report_filename: str = "metrics_report.txt"
    roc_fig_filename: str = "ROC_curve.pdf"
    pr_fig_filename: str = "PR_curve.pdf"
    fixed_threshold: float = 0.5

    # --- concept analysis  ---
    concept_out_subdir: str = "interpret_artif"
    concept_out_basename: str = "senn_mdd_interp"
    concept_top_k_channels: int = 15
    concept_top_k_presence: int = 30
    concept_min_fold_count: int = 1

    # --- motif analysis ---
    motif_num_folds: int  = None
    motif_keep_frac: float = 0.7               # keep features with r >= keep_frac * max_r per concept
    motif_out_subdir: str = "interpret_artif"  # under save_dir
    motif_top_n_groups: int = 10
    motif_top_k_tugofwar: int = 15

    # --- local inference ---
    infer_epoch_sec: float = 2.0
    infer_overlap_sec: float = 0.5
    infer_agg_method: str = "trimmed"
    infer_trim_p: float = 0.1
    infer_confidence_weighting: bool = False
    infer_top_concepts: int = 5
    infer_fold_idx: int = 0                 # which fold index to use for concept_to_motifs_full.csv
    infer_fold_dirname: str = "fold_0"      # default fold directory under save_dir
    infer_interpret_subdir: str = "interpret_artif"





def ensure_save_dir(cfg: Config) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
