"""Defaults from the revised baseline notebook executable code."""

CONFIG = {
    "seed": 42,
    "epoch_sec": 2.0,
    "epoch_overlap_sec": 0.5,
    "selector_k": 80,
    "lambda_grad": 0.02,
    "lambda_l1": 0.0005,
    "batch_size": 8,
    "lr": 1e-3,
    "epochs": 100,
    "concept_dim": 50,
    "save_dir": "./artif_senn_mdd_rev_baseline",
    "feature_cache_dir": "./senn_feature_cache",
    "n_folds": 5,
    "val_fraction": 0.20,
}
