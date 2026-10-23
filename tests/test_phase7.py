"""Fase 7: evaluación comparable, estado inmutable, curvas y reanudación sintética."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.figure
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import test_phase4 as fixtures
from src.check_suite import verify_suite
from src.checkpointing import capture_rng, load_resume
from src.compare_runs import comparison
from src.dataset import BreastDCEDataset
from src.experiment import create_criterion, evaluate_train, train, validate
from src.model import BreastCancerCNN


class Phase7Tests(unittest.TestCase):
    def test_all_train_without_augmentation_weights_bn_gradients_or_rng_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixtures.Phase4Tests().fixture(Path(tmp))
            cfg.update(USE_AUGMENTATION=True, AUGMENTATION_MODE='gentle_affine',
                       USE_WEIGHTED_LOSS=True, BATCH_SIZE=3, MAX_VAL_BATCHES=None)
            ds = BreastDCEDataset(cfg['TRAIN_CSV'], cfg['DATA_DIR'], config=cfg, augment=True)
            model = BreastCancerCNN(cfg).train()
            model(torch.randn(4, 3, 16, 16)).sum().backward()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            gradients = [p.grad.clone() for p in model.parameters()]
            generator = torch.Generator().manual_seed(42)
            rng = capture_rng(generator)
            criterion = create_criterion(cfg, torch.device('cpu'))
            # Cualquier augmentation aquí sería un error, incluso si conserva etiquetas.
            with patch('src.augmentation.synchronized_affine', side_effect=AssertionError('augmentation')):
                result = evaluate_train(model, ds, criterion, torch.device('cpu'), cfg, generator)
            self.assertTrue(ds.augment)
            self.assertTrue(model.training)
            for k, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)
            for old, p in zip(gradients, model.parameters()):
                torch.testing.assert_close(old, p.grad, rtol=0, atol=0)
            after_rng = capture_rng(generator)
            self.assertEqual(rng['python'], after_rng['python'])
            self.assertEqual(rng['numpy'], after_rng['numpy'])
            for key in ['torch', 'loader']:
                torch.testing.assert_close(rng[key], after_rng[key], rtol=0, atol=0)
            deterministic = BreastDCEDataset(cfg['TRAIN_CSV'], cfg['DATA_DIR'], config=cfg)
            expected = validate(model, DataLoader(deterministic, batch_size=3), criterion,
                                torch.device('cpu'), cfg)
            self.assertEqual(len(result['slice_predictions']), len(ds))
            self.assertFalse(result['diagnostic_partial'])
            self.assertEqual(result['loss'], expected['loss'])
            self.assertEqual(result['patient'], expected['patient'])

    def test_comparable_curves_checkpoints_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixtures.Phase4Tests().fixture(root)
            cfg.update(EVALUATE_TRAIN=True, USE_AUGMENTATION=True, AUGMENTATION_MODE='gentle_affine')
            figures = {}
            original = matplotlib.figure.Figure.savefig

            def savefig(fig, name, *args, **kwargs):
                figures[Path(name).name] = [(line.get_label(), np.array(line.get_ydata()))
                                            for line in fig.axes[0].lines]
                return original(fig, name, *args, **kwargs)

            with patch.object(matplotlib.figure.Figure, 'savefig', new=savefig):
                complete, _ = train(cfg)
            history = pd.read_csv(complete/'history.csv')
            self.assertEqual(len(figures['loss_curve.png']), 3)
            self.assertEqual(len(figures['loss_comparison.png']), 2)
            for name, columns in {
                    'loss_curve.png': ['train_loss', 'train_eval_loss', 'val_loss'],
                    'loss_comparison.png': ['train_eval_loss', 'val_loss'],
                    'accuracy_curve.png': ['train_eval_patient_accuracy', 'val_patient_accuracy']}.items():
                for (_, values), column in zip(figures[name], columns):
                    np.testing.assert_array_equal(values, history[column])
            self.assertIn('eval', figures['auc_curve.png'][0][0])
            np.testing.assert_array_equal(figures['auc_curve.png'][0][1], history.train_eval_patient_auc)
            for name in ['loss_curve', 'loss_comparison', 'accuracy_curve', 'auc_curve', 'roc_curve']:
                self.assertGreater((complete/(name+'.png')).stat().st_size, 1000)
            for epoch in [1, 2]:
                cp = torch.load(complete/f'epoch_{epoch:03d}.pt', weights_only=True)
                self.assertIsNotNone(cp['train_eval_metrics'])
                self.assertTrue(cp['optimizer_state_dict']['state'])
                self.assertIsNotNone(cp['scheduler_state_dict'])
                self.assertIn('rng_state', cp)
            self.assertTrue((complete/'best_epoch.pt').is_file())
            first, _ = train(dict(cfg, EPOCHS=1))
            resumed, _ = train(cfg, resume_from=first/'epoch_001.pt')
            a = torch.load(complete/'epoch_002.pt', weights_only=True)
            b = torch.load(resumed/'epoch_002.pt', weights_only=True)
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual(a['scheduler_state_dict'], b['scheduler_state_dict'])
            for i, state in a['optimizer_state_dict']['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b['optimizer_state_dict']['state'][i][key], rtol=0, atol=0)
            self.assertEqual(a['train_eval_metrics'], b['train_eval_metrics'])
            frame = comparison(cfg['RESULTS_DIR'])
            self.assertIn('Train_eval_loss_mejor_epoca', frame)

    def test_observation_preserves_training_trajectory_and_legacy_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixtures.Phase4Tests().fixture(Path(tmp))
            old, _ = train(dict(cfg, EVALUATE_TRAIN=False))
            new, _ = train(dict(cfg, EVALUATE_TRAIN=True))
            a = torch.load(old/'epoch_002.pt', weights_only=True)
            b = torch.load(new/'epoch_002.pt', weights_only=True)
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=0, atol=0)
            torch.testing.assert_close(a['rng_state']['torch'], b['rng_state']['torch'], rtol=0, atol=0)
            self.assertNotIn('train_eval_loss', pd.read_csv(old/'history.csv'))
            self.assertFalse((old/'loss_comparison.png').exists())
            # Contrato auténtico del commit previo: flag ausente, fuentes exactas auditadas.
            cp = torch.load(old/'epoch_001.pt', weights_only=True)
            migration = json.loads(Path('src/layout_compatibility.json').read_text(encoding='utf-8'))
            previous = migration['compatible_contracts'][-1]['sources']
            cp['training_contract']['source_sha256'] = previous
            cp['training_contract'].pop('EVALUATE_TRAIN')
            cp['config'].pop('EVALUATE_TRAIN')
            legacy = Path(tmp)/'legacy.pt'
            torch.save(cp, legacy)
            self.assertEqual(load_resume(legacy, cfg)['epoch'], 1)
            with self.assertRaisesRegex(ValueError, 'EVALUATE_TRAIN'):
                load_resume(legacy, dict(cfg, EVALUATE_TRAIN=True))

    def test_four_configurations_and_finite_synthetic_networks(self):
        result = verify_suite('config/phase7/suite.json', device='cpu', synthetic=True)
        self.assertEqual(len(result['models']), 4)
        parameters = {r['id']: r['parameters'] for r in result['models']}
        self.assertEqual(parameters['SPATIAL_CONTROL'], 288033)
        self.assertLess(parameters['SPATIAL_SMALL'], parameters['SPATIAL_CONTROL'])
        self.assertEqual(result['partitions']['train']['slices'], 8745)


if __name__ == '__main__':
    unittest.main()
