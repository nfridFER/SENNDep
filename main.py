"""Run the SENN training → evaluation → optional inference pipeline."""
import argparse
import json
from pathlib import Path
import sys

from src.config import CONFIG


def _training_arguments(parser):
    parser.add_argument("--data_folder", required=True, help="Folder containing EEG .set files.")
    parser.add_argument("--excel_path", required=True, help="Excel metadata with REC_ID and DIAGNOSIS.")
    for name, default in CONFIG.items():
        parser.add_argument(f"--{name}", type=type(default), default=default)


def _inference_arguments(parser, *, pipeline=False):
    parser.add_argument("--set_path", required=not pipeline, help="EEGLAB recording to predict.")
    parser.add_argument("--fold", type=int, default=0, help="Saved fold model to use (default 0).")
    parser.add_argument("--tau", type=float, default=0.70)
    parser.add_argument("--inference_batch_size" if pipeline else "--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--output_json", default=None, help="Save prediction and full additive explanations.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pipeline = commands.add_parser("pipeline", help="Train and evaluate, then optionally explain and infer.")
    _training_arguments(pipeline)
    _inference_arguments(pipeline, pipeline=True)
    pipeline.add_argument("--explain", action="store_true", help="Export all outer-test subject explanations.")
    pipeline.add_argument("--overlap", action="store_true", help="Analyze selected-feature overlap across folds.")
    pipeline.add_argument("--infer_after_train", action="store_true", help="Run inference after training; requires --set_path.")

    train = commands.add_parser("train", help="Run training and fold-level held-out evaluation only.")
    _training_arguments(train)

    report = commands.add_parser("eval", aliases=["report"], help="Report existing outer-test predictions without retraining.")
    report.add_argument("--root_dir", "--save_dir", dest="root_dir", default=CONFIG["save_dir"])
    report.add_argument("--excel_path", required=True)
    report.add_argument("--n_folds", type=int, default=None, help="Defaults to the saved run's fold count.")
    report.add_argument("--decision_threshold", type=float, default=0.5)

    infer = commands.add_parser("infer", help="Predict a recording with a saved fold; no training/Excel required.")
    infer.add_argument("--root_dir", "--save_dir", dest="root_dir", default=CONFIG["save_dir"])
    _inference_arguments(infer)

    overlap = commands.add_parser("overlap", help="Compare fold-specific selected features.")
    overlap.add_argument("--root_dir", default=CONFIG["save_dir"])
    overlap.add_argument("--n_folds", type=int, default=5)
    overlap.add_argument("--top_k", type=int, default=80)

    explain = commands.add_parser("explain", help="Export additive explanations for all outer-test subjects.")
    explain.add_argument("--root_dir", default=CONFIG["save_dir"])
    explain.add_argument("--excel_path", required=True, help="Original metadata supplies true labels.")
    explain.add_argument("--feature_cache_dir", default=None, help="Override the cache path saved in config.json.")
    explain.add_argument("--output_csv", default=None)
    explain.add_argument("--tau", type=float, default=0.70)
    explain.add_argument("--batch_size", type=int, default=256)
    explain.add_argument("--device", default=None)
    explain.add_argument("--subject_id", default=None, help="Also print this subject's leading concepts and motifs.")

    argv = list(sys.argv[1:] if argv is None else argv)
    # Accept the original main.py style without a subcommand, including infer-only.
    if argv and argv[0].startswith("-") and argv[0] not in {"-h", "--help"}:
        if "--infer_only" in argv:
            argv.remove("--infer_only")
            argv.insert(0, "infer")
        else:
            argv.insert(0, "pipeline")
    args = parser.parse_args(argv)
    if args.command == "pipeline":
        if args.infer_after_train and not args.set_path:
            parser.error("--infer_after_train requires --set_path")
        if args.output_json and not args.set_path:
            parser.error("--output_json requires --set_path")
        from src.pipeline import run_pipeline
        from src.inference import print_prediction

        result = run_pipeline(
            args.data_folder, args.excel_path, {name: getattr(args, name) for name in CONFIG},
            explain=args.explain, overlap=args.overlap, set_path=args.set_path, fold=args.fold,
            tau=args.tau, inference_batch_size=args.inference_batch_size, device=args.device,
            output_json=args.output_json,
        )
        if result["prediction"] is not None:
            print_prediction(result["prediction"])
    elif args.command == "train":
        from src.cv_fold import run_training

        run_training(args.data_folder, args.excel_path,
                     {name: getattr(args, name) for name in CONFIG})
    elif args.command in {"eval", "report"}:
        from src.reporting import summarize_cv_results

        n_folds = args.n_folds
        if n_folds is None:
            with (Path(args.root_dir) / "config.json").open(encoding="utf-8") as f:
                n_folds = json.load(f)["n_folds"]
        summarize_cv_results(args.root_dir, args.excel_path, n_folds, args.decision_threshold)
    elif args.command == "infer":
        from src.inference import predict_eeg_set, print_prediction

        result = predict_eeg_set(
            args.root_dir, args.set_path, fold=args.fold, tau=args.tau,
            batch_size=args.batch_size, device=args.device, output_json=args.output_json,
        )
        print_prediction(result)
    elif args.command == "overlap":
        from src.feature_overlap import summarize_feature_overlap

        summarize_feature_overlap(args.root_dir, args.n_folds, args.top_k)
    else:
        from src.subject_explanations import export_subject_explanations, show_subject

        explanations, _ = export_subject_explanations(
            args.root_dir, args.excel_path, feature_cache_dir=args.feature_cache_dir,
            output_csv=args.output_csv, tau=args.tau, batch_size=args.batch_size,
            device=args.device,
        )
        if args.subject_id is not None:
            show_subject(explanations, args.subject_id)


if __name__ == "__main__":
    main()
