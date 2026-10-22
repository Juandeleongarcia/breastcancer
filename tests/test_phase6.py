"""Fase 6: contrastes, presupuesto, pooling regional y recuperación sintética."""
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import test_phase4 as fixtures
from src.batch_train import resolve_job
from src.checkpointing import load_resume, experiment_key
from src.configuration import load_config
from src.evaluate import evaluate_checkpoint
from src.experiment import train
from src.model import BreastCancerCNN
from src.phase6 import audit, fits_budget, main as phase6_main


class Phase6Tests(unittest.TestCase):
    def test_five_isolated_changes_and_complete_partitions(self):
        entries = load_config('config/phase6/suite.json')['experiments']
        self.assertEqual(len(entries), 5)
        configs = {entry['id']: resolve_job(entry['config'], 'cpu') for entry in entries}
        result = audit(configs)
        self.assertEqual(result['partitions']['train']['patients'], 877)
        self.assertEqual(result['partitions']['validation']['patients'], 220)
        self.assertEqual(result['historical_reference']['source'], 'user_report_not_locally_verified')
        self.assertEqual(len(result['differences']), 5)
        for cfg in configs.values():
            self.assertEqual(cfg['PATIENCE'], 5)
            self.assertEqual(cfg['EPOCHS'], 10)
            self.assertFalse(cfg['USE_WEIGHTED_LOSS'])

    def test_budget_reserves_time_and_rejects_long_run(self):
        self.assertFalse(fits_budget(7200, 7200))
        self.assertFalse(fits_budget(7200, 9000))
        self.assertTrue(fits_budget(7200, 7140))
        self.assertFalse(fits_budget(59, 1))

    def test_campaign_does_not_train_when_no_model_fits(self):
        entries = load_config('config/phase6/suite.json')['experiments']
        configs = {entry['id']: resolve_job(entry['config'], 'cpu') for entry in entries}
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            started = time.time()
            state = dict(start_unix=started, deadline_unix=started+7200, device='cpu',
                         configuration_keys={name: experiment_key(cfg) for name, cfg in configs.items()},
                         benchmark={name: {'conservative_run_seconds': 18000} for name in configs}, results=[])
            (session/'execution.json').write_text(json.dumps(state))
            with patch('sys.argv', ['phase6', '--device', 'cpu', '--session', str(session)]), patch(
                    'src.phase6.train') as training, patch('src.phase6.completed_run', return_value=None):
                phase6_main()
                training.assert_not_called()
            frame = pd.read_csv(session/'comparison.csv')
            self.assertEqual(len(frame), 5)
            self.assertTrue(frame.epochs_completed.eq(0).all())
            self.assertTrue(frame.auc_patient.isna().all())
            self.assertTrue(frame.status.eq('no_ejecutado_presupuesto').all())

    def test_original_shapes_and_regional_features(self):
        torch.set_num_threads(2)
        old = BreastCancerCNN({'CONVS_PER_BLOCK': 2})
        self.assertEqual(sum(p.numel() for p in old.parameters()), 288033)
        regional = BreastCancerCNN({'CONVS_PER_BLOCK': 2, 'SPATIAL_GRID': 2})
        self.assertEqual(sum(p.numel() for p in regional.parameters()), 288417)
        # Exactamente las mismas claves, dimensiones y operaciones convolucionales.
        for name, tensor in old.features.state_dict().items():
            self.assertEqual(tensor.shape, regional.features.state_dict()[name].shape)
        x = torch.randn(4, 3, 32, 32)
        self.assertEqual(tuple(regional.features(x).shape), (4, 128, 2, 2))
        logits = regional(x)
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in regional.parameters()))

    def test_historical_missing_grid_remains_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixtures.Phase4Tests().fixture(root)
            directory, summary = train(dict(cfg, EPOCHS=1))
            checkpoint = torch.load(directory/'best_epoch.pt', weights_only=True)
            checkpoint['config'].pop('SPATIAL_GRID')
            old = root/'legacy.pt'
            torch.save(checkpoint, old)
            with patch('config.config.SPATIAL_GRID', 2):
                result = evaluate_checkpoint(old)
            self.assertEqual(result['patient'], summary['validation']['patient'])

    def test_regional_exact_resume_and_four_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixtures.Phase4Tests().fixture(Path(tmp))
            cfg['SPATIAL_GRID'] = 2
            whole, summary = train(cfg)
            first, _ = train(dict(cfg, EPOCHS=1))
            resumed, _ = train(cfg, resume_from=first/'epoch_001.pt')
            a = torch.load(whole/'epoch_002.pt', weights_only=True)
            b = torch.load(resumed/'epoch_002.pt', weights_only=True)
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual(a['scheduler_state_dict'], b['scheduler_state_dict'])
            for curve in ['loss', 'accuracy', 'auc', 'roc']:
                self.assertTrue((whole/f'{curve}_curve.png').is_file())
            self.assertEqual(evaluate_checkpoint(whole/'best_epoch.pt')['patient'], summary['validation']['patient'])
            with self.assertRaisesRegex(ValueError, 'SPATIAL_GRID'):
                load_resume(first/'epoch_001.pt', dict(cfg, SPATIAL_GRID=1))

    def test_regional_can_memorize_synthetic_batch(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = BreastCancerCNN({'CONV_CHANNELS': [4, 8], 'CONVS_PER_BLOCK': 2,
                                'SPATIAL_GRID': 2, 'DROPOUT': 0})
        images = torch.randn(4, 3, 16, 16)*.05
        labels = torch.tensor([0., 1., 0., 1.])
        images[labels.bool(), :, :8, :8] += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.eval()
        with torch.no_grad():
            initial = loss_fn(model(images).squeeze(1), labels).item()
        for _ in range(40):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = loss_fn(model(images).squeeze(1), labels).item()
        self.assertLess(final, initial*.7)


if __name__ == '__main__':
    unittest.main()
