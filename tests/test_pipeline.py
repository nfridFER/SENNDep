"""Behavioral checks for the revised notebook workflow, without private data."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.cv_fold import run_training
from src.data import EpochDataset, load_or_extract_patient_epoch_matrix
from src.explain import concept_feature_correlation_matrix
from src.feature_overlap import summarize_feature_overlap
from src.features import make_epoch_feature_names
from src.model import SENN, gradient_alignment_loss
from src.reporting import summarize_cv_results
from src.train import train_epoch_level
from src.subject_explanations import export_subject_explanations


class RevisedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = io.StringIO()
        self.quiet = contextlib.redirect_stdout(self.output)
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_cache_reuse_invalidation_and_corruption(self):
        source = self.root / "subject.set"
        source.write_bytes(b"source")
        companion = source.with_suffix(".fdt")
        companion.write_bytes(b"samples")
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        with patch("src.data.extract_patient_epoch_matrix", return_value=(features, ["F3"])) as extract:
            def load(**kwargs):
                return load_or_extract_patient_epoch_matrix(str(source), "subject", self.root / "cache", **kwargs)

            first, channels = load()
            second, _ = load()
            np.testing.assert_array_equal(first, second)
            self.assertEqual(channels, ["F3"])
            self.assertEqual(extract.call_count, 1)
            companion.write_bytes(b"changed companion samples")
            load()
            self.assertEqual(extract.call_count, 2)
            source.write_bytes(b"changed set")
            load()
            self.assertEqual(extract.call_count, 3)
            load(epoch_sec=1)
            self.assertEqual(extract.call_count, 4)
            with patch("src.data.FEATURE_CACHE_VERSION", 2):
                load(epoch_sec=1)
            self.assertEqual(extract.call_count, 5)
            (self.root / "cache/subject.npz").write_bytes(b"corrupt")
            load()
            self.assertEqual(extract.call_count, 6)
        self.assertFalse(list((self.root / "cache").glob("*.tmp.npz")))

    def test_training_uses_plain_mean_bce_even_when_weights_supplied(self):
        torch.manual_seed(42)
        model = SENN(4, concept_dim=3)
        expected = copy.deepcopy(model)
        X = np.arange(16, dtype=np.float32).reshape(4, 4) / 10
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        sids = np.array([0, 0, 1, 1])
        ds = EpochDataset(X, y, sids, weights=np.array([0.001, 7, 2, 0.03]))
        optimizer = torch.optim.AdamW(expected.parameters(), lr=0.001)
        logits, theta, concepts = expected(ds.X)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, ds.y)
                + 0.0005 * theta.abs().mean()
                + 0.02 * gradient_alignment_loss(logits, theta, concepts))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_level(model, DataLoader(ds, batch_size=4), ds, sids,
                          str(self.root), epochs=1, device="cpu")
        for actual, wanted in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        self.assertTrue((self.root / "best_model.pt").is_file())

    def test_complete_correlations_preserve_sign_order_and_notebook_normalization(self):
        X = np.array([[0, 6, 1], [1, 4, 1], [2, 2, 1], [3, 0, 1]], dtype=np.float32)
        z = np.column_stack([X[:, 0], X[:, 1]])
        ds = EpochDataset(X, np.array([0, 0, 1, 1]), np.arange(4))
        correlations = concept_feature_correlation_matrix(ds, z)
        scale = np.sqrt(3 / 4)  # torch sample SD vs NumPy population SD
        np.testing.assert_allclose(correlations, [[scale, -scale, 0], [-scale, scale, 0]], atol=1e-6)
        self.assertEqual(correlations.dtype, np.float32)

    def test_outer_cv_artifacts_preprocessing_reporting_and_overlap(self):
        channels = ["F3", "F4", "C3", "C4", "P3", "P4"]
        names = make_epoch_feature_names(channels)
        rng = np.random.default_rng(11)
        ids, diagnoses, matrices = [], [], {}
        for group, diagnosis in [("M", "MDD"), ("GA", "healthy"), ("H", "healthy")]:
            for i in range(10):
                sid = f"{group}{i:02d}"
                ids.append(sid)
                diagnoses.append(diagnosis)
                (self.root / f"{sid}.set").write_bytes(b"synthetic")
                matrices[sid] = rng.normal(size=(4, len(names))).astype(np.float32) + (3 if diagnosis == "MDD" else 0)
        excel = self.root / "labels.xlsx"
        pd.DataFrame({"REC_ID": ids, "DIAGNOSIS": diagnoses}).to_excel(excel, index=False)
        artifacts = self.root / "artifacts"
        cache = self.root / "cache"
        config = {"save_dir": str(artifacts), "feature_cache_dir": str(cache),
                  "epochs": 1, "batch_size": 32, "concept_dim": 5}
        def extract(path, **kwargs):
            return matrices[Path(path).stem], channels

        with patch("src.data.extract_patient_epoch_matrix", side_effect=extract) as extraction:
            results = run_training(str(self.root), str(excel), config)
        self.assertEqual(len(results), 5)
        self.assertEqual(extraction.call_count, 30)
        all_tests = []
        for fold in range(5):
            folder = artifacts / f"fold_{fold}"
            manifest = json.loads((folder / "split_subject_ids.json").read_text())
            train, val, test = [set(manifest[f"{split}_ids"]) for split in ("train", "val", "test")]
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(ids))
            all_tests.extend(test)
            X = np.vstack([matrices[sid] for sid in manifest["train_ids"]])
            expected_scaler = StandardScaler().fit(X)
            scaler = joblib.load(folder / "scaler.joblib")
            np.testing.assert_allclose(scaler.mean_, expected_scaler.mean_)
            np.testing.assert_allclose(scaler.scale_, expected_scaler.scale_)
            selected = np.load(folder / "selected_features.npy", allow_pickle=True)
            self.assertEqual(len(selected), 80)
            for split in ("train", "test"):
                corr = np.load(folder / f"concept_feature_corr_{split}.npy")
                self.assertEqual(corr.shape, (5, 80))
                self.assertTrue(np.isfinite(corr).all())
            for split, wanted in [("val", val), ("test", test)]:
                predictions = np.load(folder / f"{split}_subject_probs_with_confidence.npy", allow_pickle=True)
                self.assertEqual(set(predictions[:, 0]), wanted)
                p = predictions[:, 1].astype(float)
                np.testing.assert_allclose(predictions[:, 2].astype(float), 2 * np.abs(p - 0.5))
            for removed in ("concept_labels.txt", "concept_labels.npy", "subject_explanations.txt", "subject_summaries.npy"):
                self.assertFalse((folder / removed).exists())
        self.assertEqual(len(all_tests), len(set(all_tests)))
        self.assertEqual(set(all_tests), set(ids))
        with patch("matplotlib.pyplot.show"):
            summarize_cv_results(artifacts, excel)
        report = (artifacts / "metrics_report.txt").read_text()
        self.assertIn("average_precision", report)
        self.assertIn("POOLED OUT-OF-FOLD", report)
        self.assertNotIn("optimal", report)
        for name in ("ROC_curve.pdf", "PR_curve.pdf"):
            self.assertTrue((artifacts / name).read_bytes().startswith(b"%PDF"))
        summarize_feature_overlap(artifacts)
        overlaps = pd.read_csv(artifacts / "feature_selection_pairwise_jaccard.csv")
        self.assertEqual(len(overlaps), 10)
        self.assertTrue(overlaps.jaccard.between(0, 1).all())
        np.testing.assert_allclose(overlaps.jaccard, overlaps.intersection_size / overlaps.union_size)
        explanations, checks = export_subject_explanations(artifacts, excel, device="cpu")
        self.assertEqual(set(checks.subject_id), set(ids))
        self.assertEqual(len(checks), 30)
        self.assertEqual(set(explanations.record_type), {"concept_feature", "motif"})
        self.assertLess(checks.probability_error.max(), 1e-5)
        self.assertLess(checks.motif_total_error.max(), 1e-5)
        self.assertTrue((artifacts / "interpretability_analysis/subject_explanations_c5_tau0p7.csv").is_file())

    def _report_fixture(self):
        labels = self.root / "labels.xlsx"
        pd.DataFrame({"REC_ID": ["A", "B", "C", "D"],
                      "DIAGNOSIS": ["healthy", "MDD", "healthy", "MDD"]}).to_excel(labels, index=False)
        for fold, rows in enumerate([[("A", 0.1), ("B", 0.4)], [("C", 0.6), ("D", 0.9)]]):
            folder = self.root / f"fold_{fold}"
            folder.mkdir()
            np.save(folder / "test_subject_probs.npy", np.array(rows, dtype=object))
        return labels

    def test_report_distinguishes_pr_auc_from_ap_and_keeps_threshold(self):
        labels = self._report_fixture()
        with patch("matplotlib.pyplot.show"):
            summarize_cv_results(self.root, labels, n_folds=2)
        text = (self.root / "metrics_report.txt").read_text()
        pooled = text.split("POOLED OUT-OF-FOLD SUBJECTS =====")[1]
        values = {key.strip(): value.strip() for line in pooled.splitlines() if ":" in line
                  for key, value in [line.split(":", 1)]}
        y, scores = [0, 1, 0, 1], [0.1, 0.4, 0.6, 0.9]
        precision, recall, _ = precision_recall_curve(y, scores)
        self.assertEqual(float(values["pr_auc"]), round(auc(recall[::-1], precision[::-1]), 3))
        self.assertEqual(float(values["average_precision"]), round(average_precision_score(y, scores), 3))
        self.assertEqual(float(values["acc"]), 0.5)

    def test_report_rejects_missing_overlapping_and_invalid_test_outputs(self):
        labels = self._report_fixture()
        path = self.root / "fold_1/test_subject_probs.npy"
        for rows, message in [([("A", 0.3), ("D", 0.8)], "more than one"),
                              ([("C", float("nan")), ("D", 0.8)], "Non-finite"),
                              ([("C", 1.1), ("D", 0.8)], "outside"),
                              ([("C", 0.1), ("C", 0.8)], "Duplicate"),
                              ([("UNKNOWN", 0.1), ("D", 0.8)], "unknown")]:
            np.save(path, np.array(rows, dtype=object))
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                summarize_cv_results(self.root, labels, n_folds=2)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            summarize_cv_results(self.root, labels, n_folds=2)

    def test_overlap_rejects_duplicate_feature_names(self):
        folder = self.root / "fold_0"
        folder.mkdir()
        np.save(folder / "selected_features.npy", np.array(["same"] * 80))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_feature_overlap(self.root)


if __name__ == "__main__":
    unittest.main()
