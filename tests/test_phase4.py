import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.configuration import resolve_config
from src.model import BreastCancerCNN
from src.experiment import train
from src.checkpointing import load_resume, experiment_key
from src.batch_train import completed_run
from src.batch_train import main as batch_main, interrupted_checkpoint
from src.resume_validation import validate_resume_environment


class Phase4Tests(unittest.TestCase):
    def test_resume_version_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"checkpoints").mkdir()
            (root/"details").mkdir()
            checkpoint=root/"epoch_001.pt"
            environment=root/"details/environment.json"
            environment.write_text(json.dumps({"packages":{"torch":"99.0.0"}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"PyTorch incompatible"):
                validate_resume_environment(checkpoint)
            environment.write_text(json.dumps({"packages":{"torch":str(torch.__version__)}}),encoding="utf-8")
            validate_resume_environment(checkpoint)

    def fixture(self, root):
        for phase, value in [("PRE", 40), ("EARLY", 100), ("LATE", 80)]:
            Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(root/f"{phase}.png")
        for part in ["train", "validation"]:
            pd.DataFrame([dict(sample_id=f"{part}_{i}", patient_id=f"{part}_{i}", pCR=i%2, split="train",
                path_pre="PRE.png", path_early="EARLY.png", path_late="LATE.png") for i in range(4)]).to_csv(root/f"{part}.csv", index=False)
        return resolve_config(dict(TRAIN_CSV=str(root/"train.csv"), VAL_CSV=str(root/"validation.csv"),
            DATA_DIR=str(root), RESULTS_DIR=str(root/"runs"), IMAGE_SIZE=16, CONV_CHANNELS=[2,4],
            NUM_WORKERS=0, BATCH_SIZE=4, EPOCHS=2, DEVICE="cpu", RUN_NAME="unit_resume", USE_EARLY_STOPPING=False,
            SCHEDULER="plateau", SCHEDULER_PATIENCE=0))

    def test_baseline_and_spatial_gradients(self):
        torch.set_num_threads(2)
        baseline = BreastCancerCNN()
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 93825)
        historical = Path("models/RUN_20261020_170316_P3_01_1156db58ca9d/checkpoints/best_epoch.pt")
        if historical.exists():
            old = torch.load(historical, weights_only=True, map_location="cpu")
            baseline.load_state_dict(old["model_state_dict"], strict=True)
        self.assertEqual(tuple(baseline.features[0].weight.shape), (32,3,3,3))
        spatial = BreastCancerCNN({"CONVS_PER_BLOCK":2})
        result = spatial(torch.randn(2,3,32,32))
        self.assertEqual(tuple(result.shape), (2,1))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in spatial.parameters()))
        self.assertEqual(sum(p.numel() for p in spatial.parameters()), 288033)

    def test_resume_exact_weights_optimizer_scheduler_rng_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.fixture(root)
            complete,_=train(cfg)
            first,_=train(dict(cfg,EPOCHS=1))
            source=first/"epoch_001.pt"
            with self.assertRaisesRegex(ValueError,"DROPOUT"):
                train(dict(cfg,DROPOUT=.2), resume_from=source)
            resumed,summary=train(cfg,resume_from=source)
            a=torch.load(complete/"epoch_002.pt",weights_only=True)
            b=torch.load(resumed/"epoch_002.pt",weights_only=True)
            for key in a["model_state_dict"]:
                torch.testing.assert_close(a["model_state_dict"][key],b["model_state_dict"][key],rtol=0,atol=0)
            self.assertEqual(a["scheduler_state_dict"],b["scheduler_state_dict"])
            self.assertEqual(a["optimizer_state_dict"]["param_groups"],b["optimizer_state_dict"]["param_groups"])
            for index,states in a["optimizer_state_dict"]["state"].items():
                for key,value in states.items():
                    torch.testing.assert_close(value,b["optimizer_state_dict"]["state"][index][key],rtol=0,atol=0)
            torch.testing.assert_close(a["rng_state"]["torch"],b["rng_state"]["torch"],rtol=0,atol=0)
            torch.testing.assert_close(a["rng_state"]["loader"],b["rng_state"]["loader"],rtol=0,atol=0)
            self.assertEqual(summary["resume_epoch"],1)
            self.assertEqual(summary["epochs_executed_here"],1)
            self.assertNotEqual(first,resumed)
            self.assertEqual(summary["experiment_key"],experiment_key(cfg))
            self.assertIsNotNone(completed_run(cfg))
            self.assertIsNone(completed_run(dict(cfg,EPOCHS=10)))

    def test_batch_logs_errors_skips_completed_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.fixture(root);cfg["EPOCHS"]=1
            good=root/"good.json";bad=root/"bad.json"
            good.write_text(json.dumps(cfg),encoding="utf-8")
            bad.write_text(json.dumps(dict(cfg,OPTIMIZER="invalid")),encoding="utf-8")
            suite=root/"suite.json"
            suite.write_text(json.dumps(dict(experiments=[dict(id="bad",config=str(bad)),
                dict(id="good",config=str(good)),dict(id="repeat",config=str(good))])),encoding="utf-8")
            from src.configuration import project_path as real_path
            with patch("sys.argv",["batch_train","--suite",str(suite),"--device","cpu"]), patch(
                    "src.batch_train.project_path",side_effect=lambda p: root/"batches" if str(p)=="models" else real_path(p)):
                batch_main()
            batch=next((root/"batches").glob("BATCH_*"))
            results=json.loads((batch/"results.json").read_text())
            self.assertEqual([r["status"] for r in results],["error","entrenamiento_completo","omitido_completo"])
            self.assertTrue((batch/"errors.jsonl").exists())
            self.assertEqual(len(pd.read_csv(batch/"comparison.csv")),3)
            self.assertEqual(pd.read_csv(batch/"comparison.csv").Estado.tolist(),["error","entrenamiento_completo","omitido_completo"])

    def test_interrupted_uses_boundary_not_partial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            import time
            root=Path(tmp);cfg=self.fixture(root)
            directory,_=train(dict(cfg,DEADLINE_UNIX=time.time()-1))
            with self.assertRaises(ValueError):
                load_resume(directory/"internals/interrupted_epoch_001.pt",cfg)
            candidate=interrupted_checkpoint(cfg)
            self.assertEqual(candidate.name,"recovery_epoch_start.pt")


if __name__ == "__main__":
    unittest.main()
