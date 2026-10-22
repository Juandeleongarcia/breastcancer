import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.batch_train import resolve_job
from src.configuration import load_config, project_path as real_path
from src.phase6_cpu3 import cpu_config, main, PRIORITY


class CPU3Tests(unittest.TestCase):
    def test_only_runtime_epoch_cap_name_and_interpretation_change(self):
        for entry in load_config('config/phase6/suite.json')['experiments']:
            if entry['id'] not in PRIORITY:
                continue
            original = resolve_job(entry['config'], 'cpu')
            derived = cpu_config(entry)
            self.assertEqual({k for k in original if original[k] != derived[k]},
                             {'EPOCHS', 'RUN_NAME', 'INTERPRETATION'})
            self.assertEqual(original['EPOCHS'], 10)
            self.assertEqual(derived['EPOCHS'], 3)
            self.assertFalse(derived['SMOKE_TEST'])
            self.assertIsNone(derived['MAX_VAL_BATCHES'])

    def test_no_run_when_benchmark_exceeds_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sys.argv', ['phase6_cpu3', '--max-minutes', '120']), patch(
                    'src.phase6_cpu3.project_path', side_effect=lambda p: root if str(p)=='models' else real_path(p)), patch(
                    'src.phase6_cpu3.measure', return_value={'conservative_run_seconds': 100000}), patch(
                    'src.phase6_cpu3.completed_run', return_value=None), patch('src.phase6_cpu3.train') as training:
                main()
                training.assert_not_called()
            session = next(root.glob('PHASE6_CPU3_*'))
            table = pd.read_csv(session/'comparison.csv')
            self.assertEqual(table.id.tolist()[:3], PRIORITY)
            self.assertEqual(len(table), 5)
            self.assertTrue(table.epochs_completed.eq(0).all())
            self.assertTrue(table.auc_patient.isna().all())


if __name__ == '__main__':
    unittest.main()
