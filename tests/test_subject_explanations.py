"""Check subject decomposition, allocation semantics, and adapted artifact paths."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from src.eval import aggregate_subject_logits
from src.model import SENN
from src.subject_explanations import (
    analyze_subject,
    export_subject_explanations,
    motif_group,
    retained_trim_indices,
    show_subject,
)


class SubjectExplanationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "custom training outputs"
        self.cache = self.base / "moved feature cache"
        self.root.mkdir()
        self.cache.mkdir()
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def fixture(self):
        """A full C=50 checkpoint in the baseline layout, with an external cache."""
        config = {"concept_dim": 50, "n_folds": 1, "seed": 42,
                  "feature_cache_dir": "old/nonexistent/cache"}
        (self.root / "config.json").write_text(json.dumps(config))
        folder = self.root / "fold_0"
        folder.mkdir()
        ids = ["healthy01", "mdd01"]
        (folder / "split_subject_ids.json").write_text(json.dumps({"test_ids": ids}))
        features = np.array(["relpow_alpha[F3]", "sampen[F4]", "coh_alpha[F3-F4]",
                             "ratio_mean_theta_over_alpha"])
        rng = np.random.default_rng(23)
        training = rng.normal(size=(40, 4)).astype(np.float32)
        scaler = StandardScaler().fit(training)
        selector = SelectKBest(f_classif, k=4).fit(scaler.transform(training), np.arange(40) % 2)
        joblib.dump(scaler, folder / "scaler.joblib")
        joblib.dump(selector, folder / "selector.joblib")
        np.save(folder / "selected_features.npy", features)
        corr = np.tile([1.0, -0.70, 0.69, 0.20], (50, 1))
        corr[1::2] = [0.20, -0.69, -1.0, 0.70]
        np.save(folder / "concept_feature_corr_test.npy", corr)
        torch.manual_seed(17)
        model = SENN(4, 50).eval()
        torch.save(model.state_dict(), folder / "best_model.pt")
        probabilities = []
        for subject_id in ids:
            raw = rng.normal(size=(41, 4)).astype(np.float32)
            np.savez(self.cache / f"{subject_id}.npz", X_e=raw)
            x = selector.transform(scaler.transform(raw)).astype(np.float32)
            with torch.no_grad():
                logits, _, _ = model(torch.from_numpy(x))
            probability = aggregate_subject_logits(logits.numpy().ravel(), np.zeros(len(x), dtype=int))[0]
            probabilities.append((subject_id, probability))
        np.save(folder / "test_subject_probs.npy", np.array(probabilities, dtype=object))
        excel = self.base / "metadata.xlsx"
        pd.DataFrame({"REC_ID": ids, "DIAGNOSIS": ["healthy", "MDD"]}).to_excel(excel, index=False)
        return excel

    def test_trim_uses_logit_indices_and_mean_of_epoch_products(self):
        class AdditiveModel(torch.nn.Module):
            def forward(self, x):
                logits = x[:, :1]
                first = 2 * logits + torch.sin(logits)
                z = torch.cat([first, logits - first], dim=1)
                return logits, torch.ones_like(z), z

        x = np.arange(-20, 21, dtype=np.float32).reshape(-1, 1)
        x[[0, 40]] = [[-1000], [1000]]
        keep = retained_trim_indices(x.ravel())
        np.testing.assert_array_equal(keep, np.arange(2, 39))
        result = analyze_subject(AdditiveModel(), x, batch_size=3)
        with torch.no_grad():
            _, theta, z = AdditiveModel()(torch.from_numpy(x))
        expected = (theta * z).numpy()[keep].astype(np.float64).mean(axis=0)
        np.testing.assert_array_equal(result["concept_contribution"], expected)
        self.assertEqual(result["n_epochs"], 41)
        self.assertEqual(result["n_retained_epochs"], 37)
        self.assertAlmostEqual(result["subject_logit"], result["logit_from_contributions"], places=5)

    def test_exports_full_concepts_signed_arrows_motifs_and_rankings(self):
        excel = self.fixture()
        destination = self.base / "custom exports" / "subjects.csv"
        frame, checks = export_subject_explanations(
            self.root, excel, feature_cache_dir=self.cache, output_csv=destination,
            batch_size=7, device="cpu",
        )
        self.assertTrue(destination.is_file())
        self.assertEqual(set(checks.subject_id), {"healthy01", "mdd01"})
        self.assertLess(checks.probability_error.max(), 1e-5)
        self.assertLess(checks.concept_total_error.max(), 1e-5)
        self.assertLess(checks.motif_total_error.max(), 1e-5)
        cf = frame[frame.record_type == "concept_feature"]
        motifs = frame[frame.record_type == "motif"]
        for subject, rows in cf.groupby("subject_id"):
            self.assertEqual(rows.concept_id.nunique(), 50)
            self.assertEqual(len(rows), 100)  # two retained features for every concept
            np.testing.assert_allclose(rows.groupby("concept_id").allocation_weight.sum(), 1)
            self.assertTrue((rows.n_retained_epochs == 37).all())
            np.testing.assert_allclose(rows.allocated_contribution,
                                       rows.concept_contribution * rows.allocation_weight)
            contributions = rows.groupby("concept_id").allocated_contribution.sum()
            targets = rows.groupby("concept_id").concept_contribution.first()
            np.testing.assert_allclose(contributions, targets, atol=1e-12)
            expected_motifs = rows.groupby("feature_motif").allocated_contribution.sum().sort_index()
            actual = motifs[motifs.subject_id == subject].set_index("motif_name").motif_contribution.sort_index()
            np.testing.assert_allclose(actual, expected_motifs, atol=1e-12)
            ranks = rows.groupby("concept_id").concept_absolute_rank.first()
            self.assertEqual(set(ranks), set(range(1, 51)))
            subject_rows = frame[frame.subject_id == subject]
            self.assertEqual(subject_rows.record_type.iloc[0], "concept_feature")
            self.assertEqual(subject_rows.record_type.iloc[-1], "motif")
        self.assertTrue((cf.loc[cf.concept_feature_correlation < 0, "correlation_arrow"] == "↓").all())
        self.assertTrue((cf.loc[cf.concept_feature_correlation >= 0, "correlation_arrow"] == "↑").all())
        self.assertEqual(set(cf.allocation_direction), {"supporting", "opposing"})
        self.assertEqual(set(frame.true_label), {0, 1})
        self.assertEqual(set(frame.training_seed), {42})
        self.assertNotIn("training_seed_offset", frame)
        show_subject(frame, "healthy01")
        with self.assertRaises(KeyError):
            show_subject(frame, "unknown")

    def test_cli_and_moved_cache_path(self):
        excel = self.fixture()
        target = self.base / "cli.csv"
        command = [sys.executable, "main.py", "explain", "--root_dir", str(self.root),
                   "--excel_path", str(excel), "--feature_cache_dir", str(self.cache),
                   "--output_csv", str(target), "--device", "cpu", "--subject_id", "mdd01"]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Dominant motifs", result.stdout)
        self.assertTrue(target.is_file())

    def test_rejects_missing_artifacts_cache_and_probability_mismatch_before_export(self):
        excel = self.fixture()
        kwargs = {"feature_cache_dir": self.cache, "device": "cpu"}
        target = self.base / "must_not_write.csv"
        with self.assertRaisesRegex(FileNotFoundError, "cache directory"):
            export_subject_explanations(self.root, excel)
        pred_path = self.root / "fold_0/test_subject_probs.npy"
        predictions = np.load(pred_path, allow_pickle=True)
        predictions[0, 1] = 0.0
        np.save(pred_path, predictions)
        with self.assertRaisesRegex(RuntimeError, "probability check failed"):
            export_subject_explanations(self.root, excel, output_csv=target, **kwargs)
        self.assertFalse(target.exists())
        (self.cache / "healthy01.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "healthy01"):
            export_subject_explanations(self.root, excel, **kwargs)
        pred_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required artifacts"):
            export_subject_explanations(self.root, excel, **kwargs)

    def test_region_grouping_uses_feature_family_and_channel_regions(self):
        self.assertEqual(motif_group("relpow_alpha[F3]"), "LeftFrontal_AlphaPower")
        self.assertEqual(motif_group("coh_alpha[F4-F3]"), "LeftFrontalRightFrontal_Connectivity")
        self.assertEqual(motif_group("alpha_asym[F3-F4]"), "LeftFrontalRightFrontal_Asymmetry")
        self.assertEqual(motif_group("ratio_mean_theta_over_beta"), "Global_BandPowerRatio")
        with self.assertRaisesRegex(ValueError, "Unknown channel"):
            motif_group("sampen[UNKNOWN]")


if __name__ == "__main__":
    unittest.main()
