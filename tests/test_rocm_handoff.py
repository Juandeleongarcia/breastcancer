import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_phase4 as fixtures

from src.check_suite import require_rocm, verify_suite
from src.batch_train import resolve_job, main as batch_main
from src.configuration import load_config, project_path, apply_statistics, file_hash
from src.checkpointing import experiment_key, load_resume
from src.experiment import train
from src.evaluate import evaluate_checkpoint
from src.resume_validation import validate_resume_environment


class RocmHandoffTests(unittest.TestCase):
    def test_five_models_stats_and_single_interventions(self):
        result = verify_suite(synthetic=False)
        self.assertEqual(len(result['models']), 5)
        self.assertEqual(result['partitions']['train']['patients'], 877)
        self.assertEqual(result['partitions']['validation']['patients'], 220)
        self.assertFalse(result['clinical_training_performed'])
        self.assertEqual(result['models'][0]['changes'], [])
        entries = load_config('config/phase6/rocm20_suite.json')['experiments']
        for entry in entries:
            cfg = resolve_job(entry['config'], 'cpu')
            self.assertEqual(cfg['EPOCHS'], 20)
            self.assertEqual(cfg['PATIENCE'], 5)
            self.assertEqual(cfg['SEED'], 42)

    def test_rocm_guard_rejects_cpu_and_nvidia(self):
        with patch.object(torch.version, 'hip', None, create=True), patch('torch.cuda.is_available', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'ROCm'):
                require_rocm()
        with patch.object(torch.version, 'hip', '7.test', create=True), patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                require_rocm()
        with patch.object(torch.version, 'hip', '7.test', create=True), patch('torch.cuda.is_available', return_value=True):
            require_rocm()  # Prueba de decisión; no simula un forward ROCm real.

    def test_linux_line_endings_preserve_stats_and_experiment_identity(self):
        cfg = resolve_job('config/phase6/rocm20/BASE_SPATIAL.json', 'cpu')
        original_key = experiment_key(cfg)
        original = project_path(cfg['TRAIN_CSV']).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp)/'train.csv'
            csv.write_bytes(original.replace(b'\r\n', b'\n'))
            portable = dict(cfg, TRAIN_CSV=str(csv))
            apply_statistics(portable, cfg['STATS_PATH'])
            self.assertEqual(experiment_key(portable), original_key)
            csv.write_bytes(csv.read_bytes().replace(b'ISPY1_1001_z012', b'ISPY1_1001_z999', 1))
            with self.assertRaisesRegex(ValueError, 'estadísticas'):
                apply_statistics(portable, cfg['STATS_PATH'])

    def test_resume_and_evaluation_accept_only_line_ending_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixtures.Phase4Tests().fixture(root)
            directory, summary = train(dict(cfg, EPOCHS=1))
            cp = torch.load(directory/'epoch_001.pt', weights_only=True)
            for role, key in [('train', 'TRAIN_CSV'), ('validation', 'VAL_CSV')]:
                path = Path(cfg[key])
                lf = path.read_bytes().replace(b'\r\n', b'\n')
                path.write_bytes(lf.replace(b'\n', b'\r\n'))
                cp['training_contract']['csv_sha256'][role] = file_hash(path)
                cp['csv_sha256'][role] = file_hash(path)
                path.write_bytes(lf)
            legacy = root/'legacy_crlf.pt'
            torch.save(cp, legacy)
            self.assertEqual(load_resume(legacy, cfg)['epoch'], 1)
            self.assertEqual(evaluate_checkpoint(legacy)['patient'], summary['validation']['patient'])
            path = Path(cfg['TRAIN_CSV'])
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'csv_sha256'):
                load_resume(legacy, cfg)

    def test_rocm_suite_cannot_accidentally_train_on_cpu(self):
        with patch('sys.argv', ['batch_train', '--suite', 'config/phase6/rocm20_suite.json', '--device', 'cpu']), patch.object(
                torch.version, 'hip', None, create=True), patch('src.batch_train.train') as training:
            with self.assertRaisesRegex(RuntimeError, 'ROCm'):
                batch_main()
            training.assert_not_called()

    def test_resume_rejects_backend_change_and_hip_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'internals').mkdir()
            path = root/'epoch_001.pt'
            env = root/'internals/environment.json'
            info = {'packages': {'torch': str(torch.__version__)}, 'device': 'cuda', 'backend': 'cuda'}
            env.write_text(json.dumps(info))
            with patch.object(torch.version, 'hip', '7.test', create=True):
                with self.assertRaisesRegex(ValueError, 'Backend incompatible'):
                    validate_resume_environment(path)
                info.update(backend='rocm', hip_build='7.test')
                env.write_text(json.dumps(info))
                validate_resume_environment(path)
                info['hip_build'] = '6.old'
                env.write_text(json.dumps(info))
                with self.assertRaisesRegex(ValueError, 'ROCm/HIP distinto'):
                    validate_resume_environment(path)


if __name__ == '__main__':
    unittest.main()
