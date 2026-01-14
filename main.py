import os
import argparse

from src.config import Config, ensure_save_dir
from src.utils import set_seed
from src.data import (
    load_labels,
    build_id_to_file_and_label,
    make_balanced_fold_val_ids,
    print_fold_stats,
)
from src.cv_fold import train_and_eval_fold
from src.reporting import summarize_cv_results



def _print_inference_result(demo: dict):
    out = demo["out"]
    print("\n=== PREDICTION ===")
    print("File:", out["set_path"])
    print(f"p(MDD) = {out['p_mdd']:.4f}   confidence={out['confidence']:.3f}")
    print("Pred:", out["pred_label"])

    for w in demo.get("warnings", []):
        print(f"\n[WARN] {w}")

    df_summary = demo.get("df_summary", None)
    if df_summary is not None:
        print("\n=== Group summary ===")
        print(df_summary)
        print("\nTop concepts (features + motifs):")
        for c in out["top_concepts"]:
            cname = c.get("concept_name")
            cname_str = f" — {cname}" if cname else ""
            print(f"\n  C{c['concept']:02d}{cname_str}: contribution={c['contribution']:+.4f}")


            feats = c.get("features", [])
            if feats:
                feats = sorted(feats, key=lambda x: x["abs_r"], reverse=True)
                print("    Feature correlations:")
                for f in feats:
                    print(f"      {f['sign']} {f['feature']}  |r|={f['abs_r']:.3f}")
            else:
                print("    Feature correlations: (none)")

        saved = demo["paths"].get("saved_group_summary_csv", None)
        if saved:
            print(f"\nSaved: {saved}")


def main():
     parser = argparse.ArgumentParser(description="Train & evaluate SENN for MDD.")
    
    # Training inputs (required unless running --infer)
    parser.add_argument("--data_folder", required=True, help="Folder containing EEG .set files.")
    parser.add_argument("--excel_path", required=True, help="Excel file with labels/metadata.")
    
    # General
    parser.add_argument("--save_dir", default=None, help="Override cfg.save_dir")
    
    # Result analysis (run after training)
    parser.add_argument("--concept_analysis", action="store_true",
                        help="After training, parse fold concept_labels.txt and produce interpretability summaries/plots.")
    parser.add_argument("--motif_analysis", action="store_true",
                        help="After training, run motif analysis (concept_labels.txt + subject_explanations.txt).")

    
    # Inference demos
    parser.add_argument("--infer_only", action="store_true",
                        help="Inference-only: run local inference + explanation and exit (no training).")
    parser.add_argument("--infer_after_train", action="store_true",
                        help="Run a local inference example after finishing training (uses produced fold artifacts).")
    parser.add_argument("--set_path", default=None, help="Path to EEGLAB .set file for inference demo.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index to use (default cfg.infer_fold_idx).")



    args = parser.parse_args()

    cfg = Config()
    if args.save_dir is not None:
        cfg = Config(**{**cfg.__dict__, "save_dir": args.save_dir})

    ensure_save_dir(cfg)
    set_seed(cfg.seed)
    
    

    # Case 1: inference-only (no training)
    if args.infer_only:
        if not args.set_path:
            raise ValueError("--infer_only requires --set_path")

        from src.inference_local import run_local_inference

        fold_idx = args.fold if args.fold is not None else cfg.infer_fold_idx
        demo = run_local_inference(cfg, set_path=args.set_path, fold_idx=fold_idx)
        _print_inference_result(demo)
        return

    # Case 2: Training path
    if not args.excel_path or not args.data_folder:
        raise ValueError("Training requires --excel_path and --data_folder (or use --infer_only for inference-only).")

    df = load_labels(args.excel_path)
    fold_val_ids = make_balanced_fold_val_ids(df, n_folds=cfg.n_folds, seed=cfg.seed)
    print_fold_stats(df, fold_val_ids)

    id_to_file, id_to_label = build_id_to_file_and_label(args.data_folder, df)

    fold_results = []
    for fold_idx, val_ids in enumerate(fold_val_ids):
        fold_results.append(train_and_eval_fold(cfg, fold_idx, val_ids, id_to_file, id_to_label))

    print(f"\n===== {cfg.n_folds}-FOLD SUBJECT-LEVEL SUMMARY =====")
    for fr in fold_results:
        print(f"Fold {fr['fold']}: acc={fr['acc']:.3f}, roc={fr['roc']:.3f}, pr={fr['pr']:.3f}")
    print("Artifacts per fold at:", os.path.abspath(cfg.save_dir))

    summarize_cv_results(cfg, root_dir=cfg.save_dir, excel_path=args.excel_path)

    # Optional analysis over saved fold artifacts
    if args.concept_analysis:
        from src.concept_analysis import run_concept_analysis
        run_concept_analysis(cfg, root_dir=cfg.save_dir)

    if args.motif_analysis:
        from src.motif_analysis import run_motif_analysis
        run_motif_analysis(cfg, root_dir=cfg.save_dir)

    # Inference demo AFTER training (uses artifacts produced above)
    if args.infer_after_train:
        if not args.set_path:
            raise ValueError("--infer_after_train requires --set_path")

        from src.inference_local import run_local_inference

        fold_idx = args.fold if args.fold is not None else cfg.infer_fold_idx
        demo = run_local_inference(cfg, set_path=args.set_path, fold_idx=fold_idx)
        _print_inference_result(demo)


if __name__ == "__main__":
    main()

