# SENN model for EEG-based MDD detection

The pipeline follows the executable code in `Train&Eval_rev_baseline.ipynb`:
outer-test cross-validation with an inner validation holdout, raw feature caching,
complete signed concept-feature correlations, outer-test reporting, and
feature-selection overlap analysis.

>  ### ⚠️ **IMPORTANT DISCLAIMER: This is NOT production software**.
>
>**This codebase is provided as-is:**
> - it is NOT production-grade
> - it has NOT been audited for clinical, medical, or safety use
> - it makes NO guarantees regarding correctness, robustness, or suitability
> - it is NOT approved for diagnostic or clinical decision-making
>
>
> ### **The authors accept no liability for any use, misuse, or consequences arising from this code or derived results.**
>
>
> The **intended use** of this codebase:
>- understanding SENN-style interpretability
>- EEG feature engineering examples
>- complete signed concept-feature association analysis
>- reproducible academic experimentation




## Installation

Use Python 3.11 and a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU-only PyTorch, install its pinned build before the remaining requirements:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Workflow

These commands correspond to the notebook's training, reporting, and overlap
cells. Arguments replace its hardcoded local Windows paths.

```bash
python main.py train --data_folder /path/to/eeg --excel_path /path/to/data.xlsx
python main.py report --excel_path /path/to/data.xlsx
python main.py overlap
```

Each command has `--help`. Training defaults live in `src/config.py`. Outputs
default to `./artif_senn_mdd_rev_baseline`; raw features are cached in
`./senn_feature_cache`. After changing training output paths, use matching
`--root_dir` arguments for analysis. Likewise, pass matching `--n_folds` and
`--top_k` if fold/feature counts change. Reporting and overlap run independently
of training. Set `MPLBACKEND=Agg` for headless reporting.

### Method

Metadata needs `REC_ID` and `DIAGNOSIS`. Case-insensitive `healthy` maps to 0;
all other diagnoses map to 1. File matching retains the notebook's prefix
stripping. Keep `.fdt` companions with `.set` recordings. Recordings must support
cropping to 300 seconds and share consistent EEG channel layouts.

Five outer folds stratify matched subjects by MDD, GA healthy, and older healthy
controls. Inner validation takes 20% of each outer training pool, stratified by
the same groups: approximately 64% training / 16% validation / 20% test overall.
Each stratum must have enough subjects for both split stages. Scaler and ANOVA
feature selection fit inner training epochs only. Exactly 80 features are
requested by default; recordings must yield at least that many raw features.

Training uses unweighted mean BCE, L1 relevance regularization, and gradient
alignment, matching the notebook code. Dataset weights remain available but are
not applied to BCE. Checkpoints maximize inner-validation subject ROC-AUC.
Subject predictions apply sigmoid after trimmed-mean epoch-logit aggregation.
There is no hyperparameter search or threshold optimization.

### Artifacts

Each `fold_k/` contains:

- `split_subject_ids.json` and train/validation/test subject-ID maps;
- `best_model.pt`, `scaler.joblib`, `selector.joblib`, `selected_features.npy`, and `channels.npy`;
- validation/test `*_subject_probs.npy` and `*_subject_probs_with_confidence.npy`;
- `concept_feature_corr_train.npy` and `concept_feature_corr_test.npy`, with
  concept rows and selected-feature columns.

Raw `.npz` caches contain unscaled, unselected features and channel names. Cache
validation checks source path, size, modification time, optional same-stem `.fdt`,
epoch settings, and cache version. Bump `FEATURE_CACHE_VERSION` or clear the cache
when feature definitions change.

`report` requires all test folds and writes `metrics_report.txt`, `ROC_curve.pdf`,
and `PR_curve.pdf`. It reports per-fold and pooled out-of-fold classification
metrics at the prespecified threshold (default 0.5), ROC-AUC, trapezoidal PR-AUC,
and average precision separately. Inputs are checked for duplicate/unknown IDs,
missing folds, non-finite scores, and probabilities outside [0, 1].

`overlap` validates the selected feature names and writes
`feature_selection_pairwise_jaccard.csv` and
`feature_selection_jaccard_summary.txt`. Five folds yield ten comparisons,
summarized with mean and sample SD. Fold-pair comparisons are not independent.

Concept labels, subject contribution/stability summaries, motif analysis,
topomaps, standalone inference, and F1-optimized thresholds have been removed
because they are outside the revised notebook workflow.

## Code structure

- `main.py`: train, report, and overlap commands.
- `src/data.py`, `src/features.py`: feature cache and EEG feature extraction.
- `src/model.py`, `src/train.py`, `src/eval.py`: SENN, training, and subject evaluation.
- `src/cv_fold.py`: outer CV and inner validation orchestration.
- `src/artifacts.py`, `src/explain.py`: preprocessing/maps and full correlations.
- `src/reporting.py`, `src/feature_overlap.py`: post-training analyses.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic features and temporary artifacts. They do not require the
private dataset or establish research/clinical performance.

## Dataset Disclaimer & Access

The example data parsing and feature extraction pipeline is tailored to an EEG dataset collected at [University Hospital Vrapče, Zagreb, Croatia](https://bolnica-vrapce.hr/) .

This dataset is **not** included in the repository. Researchers interested in the dataset may request access strictly for academic research purposes.

> 📧 Contact: [nikolina.frid@fer.unizg.hr](mailto:nikolina.frid@fer.unizg.ht)

Please include:
- your institution
- a short description of your research project
- the intended scientific use of the data

> ⚠️ Requests from commercial entities or projects intended for commercial exploitation will not be considered.

---

