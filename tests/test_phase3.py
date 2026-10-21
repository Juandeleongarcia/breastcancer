import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.configuration import resolve_config
from src.experiment import train
from src.evaluate import evaluate_checkpoint
from src.phase3 import plan


class Phase3Tests(unittest.TestCase):
    def test_plan_changes_one_primary_variable(self):
        entries = {entry["name"]: entry for entry in plan()}
        self.assertEqual(len(entries), 10)
        main_keys = ["INPUT_MODE", "USE_CLIPPING", "USE_STANDARDIZATION", "LEARNING_RATE",
                     "DROPOUT", "USE_WEIGHTED_LOSS", "CONV_CHANNELS"]
        for name, entry in entries.items():
            cfg = entry["config"]
            self.assertEqual(cfg["EPOCHS"], 3)
            self.assertIsNone(cfg["MAX_VAL_BATCHES"])
            if entry["comparison_parent"]:
                parent = entries[entry["comparison_parent"]]["config"]
                self.assertEqual(sum(cfg[key] != parent[key] for key in main_keys), 1, name)

    def test_new_layout_checkpoint_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase, value in [("PRE", 40), ("EARLY", 100), ("LATE", 80)]:
                Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(root / f"{phase}.png")
            for part in ["train", "validation"]:
                frame = pd.DataFrame([dict(sample_id=f"{part}_{i}", patient_id=f"{part}_{i}",
                    pCR=i%2, split="train", path_pre="PRE.png", path_early="EARLY.png", path_late="LATE.png") for i in range(4)])
                frame.to_csv(root / f"{part}.csv", index=False)
            cfg = resolve_config(dict(TRAIN_CSV=str(root/"train.csv"), VAL_CSV=str(root/"validation.csv"),
                DATA_DIR=str(root), RESULTS_DIR=str(root/"runs"), IMAGE_SIZE=16, CONV_CHANNELS=[2, 4],
                NUM_WORKERS=0, BATCH_SIZE=4, EPOCHS=1, DEVICE="cpu", RUN_NAME="unit_layout"))
            directory, summary = train(cfg)
            self.assertTrue((directory/"history.csv").exists())
            self.assertTrue((directory/"config.py").exists())
            self.assertTrue((directory/"internals/config.json").exists())
            self.assertTrue((directory/"best_epoch.pt").exists())
            self.assertFalse((directory/"checkpoints").exists())
            saved = torch.load(directory/"best_epoch.pt", weights_only=True)
            saved["config"]["DEADLINE_UNIX"] = time.time()-10
            test_checkpoint = root/"expired_deadline.pt"
            torch.save(saved, test_checkpoint)
            independent = evaluate_checkpoint(test_checkpoint)
            self.assertEqual(independent["patient"], summary["validation"]["patient"])
            cfg["DEADLINE_UNIX"] = time.time()-1
            interrupted_dir, interrupted = train(cfg)
            self.assertEqual(interrupted["epochs_completed"], 0)
            self.assertTrue((interrupted_dir/"internals/interrupted_epoch_001.pt").exists())
            self.assertFalse((interrupted_dir/"best_epoch.pt").exists())


if __name__ == "__main__":
    unittest.main()
