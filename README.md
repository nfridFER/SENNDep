# SENN model for EEG-based MDD Detection – Research Showcase

This repository contains a **research-oriented implementation of a Self-Explaining Neural Network (SENN)** for EEG-based MDD detection, with a strong focus on **interpretability** (concepts and motifs).

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
>- concept- and motif-based explanations
>- reproducible academic experimentation



---

## Code Structure (high-level)

```text
.
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Central configuration
│   ├── data.py                 # Dataset parsing + feature extraction
│   ├── model.py                # SENN model definition
│   ├── cv_fold.py              # Fold-level training & evaluation
│   ├── eval.py                 # Aggregation & evaluation utilities
│   ├── reporting.py            # CV summaries and metrics
│   ├── concept_analysis.py     # Concept analysis
│   ├── motif_analysis.py       # Motif mining & grouping
│   └── inference_local.py      # Local inference + explanations
└── README.md
```


## `requirements.txt`

Lists the **Python package dependencies** required to run the code  
(e.g. PyTorch, NumPy, pandas, SciPy, scikit-learn, MNE, etc.).

It is recommended to install them in a **virtual environment**:

```bash
pip install -r requirements.txt
```

Version pinning is intentionally minimal; this repository prioritizes research flexibility over long-term dependency stability.

---

## `main.py`

Acts purely as a **dispatcher**:

- cross-validated training & evaluation
- optional post-hoc analyses (concepts, motifs)
- two inference modes:
  - **inference-only** (using existing fold artifacts)
  - **inference after training** (demo on freshly trained models)

All actual logic lives in `src/`.

---

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

