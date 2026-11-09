"""Compose the revised training, held-out evaluation, and inference workflows."""
from pathlib import Path

from .config import CONFIG
from .cv_fold import run_training
from .feature_overlap import summarize_feature_overlap
from .inference import predict_eeg_set
from .reporting import summarize_cv_results
from .subject_explanations import export_subject_explanations


def run_pipeline(data_folder, excel_path, config=None, *, explain=False, overlap=False,
                 set_path=None, fold=0, tau=0.70, inference_batch_size=256,
                 device=None, output_json=None):
    """Train, report outer-test results, optionally explain and predict a recording."""
    config = {**CONFIG, **(config or {})}
    if set_path is not None:
        if not Path(set_path).is_file():
            raise FileNotFoundError(f"Missing EEG recording: {set_path}")
        if not 0 <= fold < int(config["n_folds"]):
            raise ValueError("Inference fold must belong to the configured training run")
    if output_json is not None and set_path is None:
        raise ValueError("An inference recording is required with output_json")
    results = run_training(data_folder, excel_path, config)
    summarize_cv_results(config["save_dir"], excel_path, n_folds=config["n_folds"])
    if overlap:
        summarize_feature_overlap(config["save_dir"], config["n_folds"], config["selector_k"])
    if explain:
        export_subject_explanations(
            config["save_dir"], excel_path, feature_cache_dir=config["feature_cache_dir"],
            tau=tau, batch_size=inference_batch_size, device=device,
        )
    prediction = None
    if set_path is not None:
        prediction = predict_eeg_set(
            config["save_dir"], set_path, fold=fold, tau=tau,
            batch_size=inference_batch_size, device=device, output_json=output_json,
        )
    return {"fold_results": results, "prediction": prediction}
