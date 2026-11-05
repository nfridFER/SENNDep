"""Run the revised notebook's training, reporting, or feature-overlap cell."""
import argparse

from src.config import CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train with outer-test CV and inner validation.")
    train.add_argument("--data_folder", required=True, help="Folder containing EEG .set files.")
    train.add_argument("--excel_path", required=True, help="Excel metadata with REC_ID and DIAGNOSIS.")
    for name, default in CONFIG.items():
        train.add_argument(f"--{name}", type=type(default), default=default)

    report = commands.add_parser("report", help="Report fixed-threshold outer-test metrics.")
    report.add_argument("--root_dir", default=CONFIG["save_dir"])
    report.add_argument("--excel_path", required=True)
    report.add_argument("--n_folds", type=int, default=CONFIG["n_folds"])
    report.add_argument("--decision_threshold", type=float, default=0.5)

    overlap = commands.add_parser("overlap", help="Compare fold-specific selected features.")
    overlap.add_argument("--root_dir", default=CONFIG["save_dir"])
    overlap.add_argument("--n_folds", type=int, default=5)
    overlap.add_argument("--top_k", type=int, default=80)

    args = parser.parse_args()
    if args.command == "train":
        from src.cv_fold import run_training

        run_training(args.data_folder, args.excel_path,
                     {name: getattr(args, name) for name in CONFIG})
    elif args.command == "report":
        from src.reporting import summarize_cv_results

        summarize_cv_results(args.root_dir, args.excel_path,
                             args.n_folds, args.decision_threshold)
    else:
        from src.feature_overlap import summarize_feature_overlap

        summarize_feature_overlap(args.root_dir, args.n_folds, args.top_k)


if __name__ == "__main__":
    main()
