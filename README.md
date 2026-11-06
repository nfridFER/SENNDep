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

### Subject explanations

After training, export additive explanations following the executable code in
`SENN_export_subject_explanations_c50_tau07.ipynb`:

```bash
python main.py explain --excel_path /path/to/data.xlsx
```

For artifacts or caches stored elsewhere:

```bash
python main.py explain --root_dir /path/to/training_outputs \
  --excel_path /path/to/data.xlsx --feature_cache_dir /path/to/senn_feature_cache \
  --output_csv /path/to/subject_explanations.csv --subject_id 4079
```

No retraining or raw EEG files are needed. The exporter uses `config.json` to
determine model size, seeds, fold count, and the feature cache path. Relative
cache paths retain their meaning relative to the training working directory;
use `--feature_cache_dir` if that directory or cache has moved. Each baseline
`fold_k/` already contains the model, scaler, selector, feature names, split IDs,
`test_subject_probs.npy`, and `concept_feature_corr_test.npy`; no ablation run or
`shared/` directory is required. Excel metadata supplies true labels because the
baseline prediction files contain `(REC_ID, probability)`, not the notebook's
`(REC_ID, true_label, probability)` format.

Every outer-test subject is explained once. Epochs are sorted by their logits,
5% are removed from each tail (rounded down), and the retained epoch products
`theta * z` are averaged directly. For each concept, features satisfying
`abs(r) >= tau * max(abs(r))` are retained; normalized absolute correlations
allocate that concept's signed contribution to features. Default `tau` is 0.7.
Correlation signs/arrows are recorded independently of contribution direction.
All concept allocations are then summed into the notebook's regional motif
groups. Positive contributions support the MDD logit; negative contributions
oppose it, regardless of the subject's predicted class. Predictions use 0.5.

The default output is
`<root_dir>/interpretability_analysis/subject_explanations_c50_tau0p7.csv`
(the model dimension and tau in the name reflect actual settings). This one
long-format CSV contains all concept-feature allocations followed by grouped
motif rows for each subject, including signs, ranks, true labels, saved
probabilities, epoch counts, and the actual fold training seed. `--subject_id`
also prints the notebook's optional single-subject view. `--batch_size` controls
inference batches (default 256); `--device cpu` forces CPU execution.

Before writing, the exporter checks that concept/motif totals reconstruct the
trimmed subject logit and its saved probability (tolerance `1e-5`) and that each
concept's feature allocations sum to its contribution (tolerance `1e-10`).
Missing artifacts/caches, duplicate test subjects, mismatched IDs, unsupported
feature/channel names, and inconsistent probabilities are reported as errors.

Legacy top-five concept-label files, epoch-stability summaries, motif/topomap
plots, standalone new-recording inference, and F1-optimized thresholds remain
outside this workflow.

## Code structure

- `main.py`: train, report, overlap, and explain commands.
- `src/data.py`, `src/features.py`: feature cache and EEG feature extraction.
- `src/model.py`, `src/train.py`, `src/eval.py`: SENN, training, and subject evaluation.
- `src/cv_fold.py`: outer CV and inner validation orchestration.
- `src/artifacts.py`, `src/explain.py`: preprocessing/maps and full correlations.
- `src/reporting.py`, `src/feature_overlap.py`: post-training analyses.
- `src/subject_explanations.py`: additive subject concept/feature/motif CSV export.

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

