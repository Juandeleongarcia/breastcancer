"""Regresiones de agregación, AUC continuo y contratos de preprocesamiento."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.configuration import resolve_config, apply_statistics
from src.dataset import BreastDCEDataset
from src.metrics import evaluate_predictions, binary_metrics
from src.model import BreastCancerCNN
from src.experiment import train


class MetricsTests(unittest.TestCase):
    def setUp(self):
        # AUC de cortes=0.50; medias de paciente: negativa=0.45, positiva=0.50.
        self.rows = [dict(sample_id=f"s{i}", patient_id=p, label=y, probability=v)
                     for i, (p, y, v) in enumerate([("n", 0, .1), ("p", 1, .4),
                                                    ("n", 0, .8), ("p", 1, .6)])]

    def test_continuous_auc_and_patient_mean(self):
        result = evaluate_predictions(self.rows)
        self.assertEqual(result["slice"]["auc"], .5)
        self.assertEqual(result["patient"]["auc"], 1)
        self.assertEqual(result["patient"]["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(result["patient"]["f1"], 1)

    def test_order_independent(self):
        first = evaluate_predictions(self.rows)
        second = evaluate_predictions(self.rows[::-1])
        self.assertEqual(first["patient"], second["patient"])
        pd.testing.assert_frame_equal(first["patient_predictions"], second["patient_predictions"])

    def test_conflicting_labels(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["label"] = 1
        with self.assertRaises(ValueError):
            evaluate_predictions(rows)

    def test_duplicate_slice_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(self.rows + [self.rows[0]])

    def test_invalid_values(self):
        for probabilities in [[np.nan, .2], [1.1, .2], [np.inf, .2]]:
            with self.assertRaises(ValueError):
                binary_metrics([0, 1], probabilities)

    def test_single_class_auc_undefined(self):
        self.assertTrue(np.isnan(binary_metrics([0, 0], [.2, .3])["auc"]))

    def test_auc_not_thresholded(self):
        self.assertEqual(binary_metrics([0, 1], [.1, .2])["auc"], 1)
        self.assertEqual(binary_metrics([0, 1], [.1, .2])["recall"], 0)

    def test_exact_confusion(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .7, .2, .8])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision"], .5)


class PipelineTests(unittest.TestCase):
    def test_stats_stage_and_phase_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase, value in [("PRE", 51), ("EARLY", 153), ("LATE", 102)]:
                Image.fromarray(np.full((256, 256), value, dtype=np.uint8)).save(root / f"{phase}.png")
            csv = root / "rows.csv"
            pd.DataFrame([dict(sample_id="s", patient_id="p", pCR=1, split="train",
                               path_pre="PRE.png", path_early="EARLY.png", path_late="LATE.png")]).to_csv(csv, index=False)
            cfg = resolve_config(dict(USE_CLIPPING=False, USE_STANDARDIZATION=True,
                                      CHANNEL_MEANS=[1., 1., 1.], CHANNEL_STDS=[2., 2., 2.]))
            raw = BreastDCEDataset(csv, root, config=cfg, standardize=False)[0]["image"]
            normalized = BreastDCEDataset(csv, root, config=cfg)[0]["image"]
            torch.testing.assert_close(raw[:, 0, 0], torch.tensor([.2, .4, .2]))
            torch.testing.assert_close(normalized, (raw - 1) / 2)

    def test_model_baseline_parameters(self):
        self.assertEqual(sum(p.numel() for p in BreastCancerCNN().parameters()), 93825)
        self.assertEqual(sum(p.numel() for p in BreastCancerCNN({"CONV_CHANNELS": [32, 64]}).parameters()), 19649)

    def test_patient_metric_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            resolve_config({"PRIMARY_METRIC": "slice_auc"})

    def test_changed_preprocessing_requires_new_statistics(self):
        with self.assertRaisesRegex(ValueError, "estadísticas"):
            train({"USE_CLIPPING": False})

    def test_partial_statistics_cannot_normalize_runs(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(json.dumps({"diagnostic_partial": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parciales"):
                apply_statistics(resolve_config(), path)


if __name__ == "__main__":
    unittest.main()
