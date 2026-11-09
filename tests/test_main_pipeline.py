"""Exercise the CLI pipeline and saved-fold inference without private data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import torch

from main import main
from src.features import make_epoch_feature_names
from src.inference import predict_eeg_set
from src.model import SENN


class MainPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "custom output"
        self.cache = self.base / "custom cache"
        self.channels = ["F3", "F4", "C3", "C4", "P3", "P4"]
        self.names = np.array(make_epoch_feature_names(self.channels))
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_default_cli_runs_training_evaluation_explanations_and_inference(self):
        rng = np.random.default_rng(10)
        matrices, ids, labels = {}, [], []
        for label, count in [("MDD", 10), ("healthy", 20)]:
            for i in range(count):
                sid = f"subject_{len(ids):03d}"
                ids.append(sid)
                labels.append(label)
                (self.base / f"{sid}.set").touch()
                matrices[sid] = (rng.normal(size=(4, len(self.names))) + (label == "MDD")).astype(np.float32)
        matrices["new_subject"] = rng.normal(size=(41, len(self.names))).astype(np.float32)
        recording = self.base / "new_subject.set"
        recording.touch()
        excel = self.base / "labels.xlsx"
        pd.DataFrame({"REC_ID": ids, "DIAGNOSIS": labels}).to_excel(excel, index=False)
        output = self.base / "prediction.json"
        def extract(path, **kwargs):
            return matrices[Path(path).stem], self.channels

        with patch("src.data.extract_patient_epoch_matrix", side_effect=extract), \
             patch("src.inference.extract_patient_epoch_matrix", side_effect=extract), \
             patch("matplotlib.pyplot.show"):
            main(["--data_folder", str(self.base), "--excel_path", str(excel),
                  "--save_dir", str(self.root), "--feature_cache_dir", str(self.cache),
                  "--epochs", "1", "--batch_size", "32", "--concept_dim", "5",
                  "--n_folds", "2", "--explain", "--overlap", "--infer_after_train",
                  "--set_path", str(recording), "--output_json", str(output), "--device", "cpu"])
            prediction = json.loads(output.read_text())
            self.assertEqual(prediction["n_retained_epochs"], 37)
            self.assertEqual(prediction["correlation_source"], "inner_training")
            self.assertTrue((self.root / "metrics_report.txt").is_file())
            self.assertTrue((self.root / "ROC_curve.pdf").is_file())
            self.assertTrue((self.root / "PR_curve.pdf").is_file())
            self.assertTrue((self.root / "feature_selection_pairwise_jaccard.csv").is_file())
            self.assertTrue((self.root / "interpretability_analysis/subject_explanations_c5_tau0p7.csv").is_file())
            self.assertAlmostEqual(sum(row["contribution"] for row in prediction["concepts"]),
                                   prediction["subject_logit"], places=5)
            self.assertAlmostEqual(sum(row["contribution"] for row in prediction["motifs"]),
                                   prediction["subject_logit"], places=5)
            for concept in prediction["concepts"]:
                self.assertAlmostEqual(sum(row["allocated_contribution"] for row in concept["features"]),
                                       concept["contribution"], places=10)
            # Evaluation gets n_folds=2 from config, rather than assuming five.
            with patch("src.pipeline.run_training", side_effect=AssertionError("must not retrain")):
                main(["eval", "--root_dir", str(self.root), "--excel_path", str(excel)])
            # The legacy infer-only form needs neither training input nor Excel.
            standalone = self.base / "standalone.json"
            main(["--infer_only", "--save_dir", str(self.root), "--set_path", str(recording),
                  "--output_json", str(standalone), "--device", "cpu"])
            self.assertEqual(json.loads(standalone.read_text()), prediction)
            with patch("src.inference.extract_patient_epoch_matrix",
                       return_value=(matrices["new_subject"], self.channels[::-1])):
                with self.assertRaisesRegex(ValueError, "channel names/order"):
                    predict_eeg_set(self.root, recording, device="cpu")

    def test_inference_reads_a_real_synthetic_eeglab_recording(self):
        self.root.mkdir()
        folder = self.root / "fold_0"
        folder.mkdir()
        config = {"n_folds": 1, "concept_dim": 5, "epoch_sec": 2.0, "epoch_overlap_sec": 0.5}
        (self.root / "config.json").write_text(json.dumps(config))
        rng = np.random.default_rng(32)
        raw_features = rng.normal(size=(20, len(self.names))).astype(np.float32)
        scaler = StandardScaler().fit(raw_features)
        selector = SelectKBest(f_classif, k=80).fit(scaler.transform(raw_features), np.arange(20) % 2)
        joblib.dump(scaler, folder / "scaler.joblib")
        joblib.dump(selector, folder / "selector.joblib")
        np.save(folder / "channels.npy", np.array(self.channels, dtype=object))
        selected = self.names[selector.get_support()]
        np.save(folder / "selected_features.npy", selected)
        np.save(folder / "concept_feature_corr_train.npy", rng.uniform(-1, 1, size=(5, 80)))
        torch.manual_seed(42)
        torch.save(SENN(80, 5).state_dict(), folder / "best_model.pt")
        # Valid continuous EEGLAB .set, with embedded microvolt data.
        sfreq = 128
        n_times = 301 * sfreq
        recording = self.base / "synthetic.set"
        eeg = {"data": rng.normal(size=(6, n_times)), "srate": sfreq,
               "nbchan": 6, "pnts": n_times, "trials": 1, "xmin": 0,
               "xmax": (n_times - 1) / sfreq,
               "chanlocs": np.array([{"labels": channel} for channel in self.channels], dtype=object)}
        savemat(recording, {"EEG": eeg}, appendmat=False)
        result = predict_eeg_set(self.root, recording, device="cpu")
        self.assertEqual(result["n_epochs"], 199)
        self.assertEqual(result["n_retained_epochs"], 181)
        self.assertTrue(0 <= result["p_mdd"] <= 1)
        self.assertEqual(len(result["concepts"]), 5)

    def test_missing_inference_argument_fails_before_training(self):
        with patch("src.pipeline.run_pipeline") as pipeline, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--data_folder", "eeg", "--excel_path", "labels.xlsx", "--infer_after_train"])
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
