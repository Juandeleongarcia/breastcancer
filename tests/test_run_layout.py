"""Contrato visual y de recuperación, usando solo cuatro imágenes sintéticas."""
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.figure
import numpy as np
import pandas as pd
import torch

import test_phase4 as fixtures
from src.batch_train import completed_run, interrupted_checkpoint
from src.checkpointing import compatible_experiment_keys, experiment_key, load_resume
from src.evaluate import evaluate_checkpoint
from src.experiment import train
from src.resume_validation import validate_resume_environment


class SimpleRunTests(unittest.TestCase):
    def test_files_curves_and_complete_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixtures.Phase4Tests().fixture(root)
            cfg['SAVE_EVERY'] = 99  # El formato nuevo conserva TODAS las épocas.
            figures = {}
            original = matplotlib.figure.Figure.savefig

            def savefig(figure, filename, *args, **kwargs):
                ax = figure.axes[0]
                figures[Path(filename).name] = dict(
                    title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                    labels=[line.get_label() for line in ax.lines],
                    values=[np.asarray(line.get_ydata()).copy() for line in ax.lines])
                return original(figure, filename, *args, **kwargs)

            with patch.object(matplotlib.figure.Figure, 'savefig', new=savefig):
                directory, summary = train(cfg)
            expected = {'config.py', 'history.csv', 'epoch_001.pt', 'epoch_002.pt',
                        'best_epoch.pt', 'loss_curve.png', 'accuracy_curve.png',
                        'auc_curve.png', 'roc_curve.png', 'internals'}
            self.assertEqual({p.name for p in directory.iterdir()}, expected)
            self.assertFalse(any(p.is_dir() for p in (directory/'internals').iterdir()))
            self.assertRegex(directory.name, r'^RUN_\d{8}_\d{6}_unit_resume_[0-9a-f]{12}$')
            exact = runpy.run_path(str(directory/'config.py'))
            saved = json.loads((directory/'internals/config.json').read_text())
            self.assertEqual(saved, {k: exact[k] for k in saved})
            history = pd.read_csv(directory/'history.csv')
            for metric in ['accuracy', 'auc']:
                figure = figures[f'{metric}_curve.png']
                self.assertIn('paciente', figure['ylabel'])
                self.assertIn('online', figure['labels'][0])
                self.assertIn('fin de época', figure['labels'][1])
                np.testing.assert_array_equal(figure['values'][0], history[f'train_patient_{metric}'])
                np.testing.assert_array_equal(figure['values'][1], history[f'val_patient_{metric}'])
            for name in ['loss_curve.png', 'accuracy_curve.png', 'auc_curve.png', 'roc_curve.png']:
                self.assertGreater((directory/name).stat().st_size, 1000)
                self.assertTrue(all(figures[name][k] for k in ['title', 'xlabel', 'ylabel']))
            self.assertIn('por corte', figures['loss_curve.png']['ylabel'])
            self.assertIn('paciente', figures['roc_curve.png']['title'])
            for epoch in [1, 2]:
                checkpoint = torch.load(directory/f'epoch_{epoch:03d}.pt', weights_only=True)
                self.assertEqual(checkpoint['epoch'], epoch)
                self.assertTrue(checkpoint['optimizer_state_dict']['state'])
                self.assertIsNotNone(checkpoint['scheduler_state_dict'])
                self.assertIn('loader', checkpoint['rng_state'])
            result = evaluate_checkpoint(directory/'best_epoch.pt')
            self.assertEqual(result['patient'], summary['validation']['patient'])
            validate_resume_environment(directory/'epoch_002.pt')

    def test_previous_layout_read_evaluate_skip_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixtures.Phase4Tests().fixture(root)
            directory, summary = train(dict(cfg, EPOCHS=1))
            old = root/'runs/RUN_20000101_000000_legacy_123456789abc'
            (old/'details').mkdir(parents=True)
            (old/'checkpoints').mkdir()
            for name in ['config.json', 'environment.json']:
                shutil.copy2(directory/'internals'/name, old/'details'/name)
            checkpoint = torch.load(directory/'epoch_001.pt', weights_only=True)
            migration = json.loads(Path('src/layout_compatibility.json').read_text())
            checkpoint['training_contract']['source_sha256'] = migration['previous_sources']
            torch.save(checkpoint, old/'checkpoints/last_epoch.pt')
            torch.save(checkpoint, old/'checkpoints/best_epoch.pt')
            torch.save(checkpoint, old/'checkpoints/epoch_001.pt')
            summary['experiment_key'] = next(k for k in compatible_experiment_keys(dict(cfg, EPOCHS=1))
                                             if k != experiment_key(dict(cfg, EPOCHS=1)))
            (old/'summary.json').write_text(json.dumps(summary))
            # Hacer visible solo el completado histórico al buscador.
            (directory/'internals/summary.json').unlink()
            self.assertEqual(completed_run(dict(cfg, EPOCHS=1)), old)
            source = old/'checkpoints/last_epoch.pt'
            validate_resume_environment(source)
            self.assertEqual(load_resume(source, cfg)['epoch'], 1)
            (old/'details/config.json').write_text(json.dumps(cfg))
            self.assertEqual(interrupted_checkpoint(cfg), old/'checkpoints/epoch_001.pt')
            result = evaluate_checkpoint(old/'checkpoints/best_epoch.pt')
            self.assertEqual(result['patient'], summary['validation']['patient'])
            resumed, _ = train(cfg, resume_from=source)
            self.assertTrue((resumed/'epoch_002.pt').is_file())
            checkpoint['training_contract']['source_sha256']['src/model.py'] = 'untrusted'
            torch.save(checkpoint, old/'checkpoints/untrusted.pt')
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                load_resume(old/'checkpoints/untrusted.pt', cfg)


if __name__ == '__main__':
    unittest.main()
