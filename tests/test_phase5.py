import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.augmentation import synchronized_affine
from src.configuration import resolve_config
from src.dataset import BreastDCEDataset
from src.experiment import train
from src.batch_train import main as batch_main
import test_phase4 as phase4_tests


class Phase5Tests(unittest.TestCase):
    def test_pending_budget_has_results_and_comparison(self):
        from src.configuration import project_path as real_path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/'suite.json'
            suite.write_text(json.dumps({'experiments':[{'id':'pending','config':'not_loaded.json'}]}),encoding='utf-8')
            with patch('sys.argv',['batch_train','--suite',str(suite),'--device','cpu','--max-minutes','.01']), patch(
                    'src.batch_train.project_path',side_effect=lambda p: root/'batches' if str(p)=='models' else real_path(p)), patch(
                    'src.batch_train.train') as training:
                batch_main();training.assert_not_called()
            batch=next((root/'batches').glob('BATCH_*'))
            self.assertEqual(json.loads((batch/'results.json').read_text())[0]['status'],'pendiente_por_presupuesto')
            self.assertTrue((batch/'comparison.csv').exists())
            self.assertEqual(len(pd.read_csv(batch/'comparison.csv')),1)
            self.assertEqual(pd.read_csv(batch/'comparison.csv').Estado.iloc[0],'pendiente_por_presupuesto')

    def test_affine_preserves_phase_relations_and_rng(self):
        cfg=resolve_config(dict(AUGMENT_PROBABILITY=1,AUGMENTATION_MODE='gentle_affine'))
        pre=torch.linspace(0,1,32*32).reshape(32,32)
        phases=torch.stack([pre,.7*pre+.1,.5*pre+.2])
        enhancement=torch.stack([phases[0],phases[1]-phases[0],phases[2]-phases[0]])
        torch.manual_seed(123);raw=synchronized_affine(phases,cfg)
        torch.manual_seed(123);enh=synchronized_affine(enhancement,cfg)
        torch.testing.assert_close(raw[1]-raw[0],enh[1],atol=2e-7,rtol=1e-5)
        torch.testing.assert_close(raw[2]-raw[0],enh[2],atol=2e-7,rtol=1e-5)
        torch.manual_seed(123);again=synchronized_affine(phases,cfg)
        torch.testing.assert_close(raw,again,atol=0,rtol=0)
        self.assertFalse(torch.equal(raw,phases))
        self.assertGreaterEqual(float(raw.min()),float(phases.min())-1e-7)
        self.assertLessEqual(float(raw.max()),float(phases.max())+1e-7)

    def test_validation_deterministic_and_training_synchronized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=phase4_tests.Phase4Tests().fixture(root)
            pattern=np.arange(256,dtype=np.uint8).reshape(16,16)
            for phase in ['PRE','EARLY','LATE']:Image.fromarray(pattern).save(root/f'{phase}.png')
            cfg.update(USE_AUGMENTATION=True,AUGMENTATION_MODE='gentle_affine',AUGMENT_PROBABILITY=1,
                       USE_CLIPPING=False,USE_STANDARDIZATION=False,INPUT_MODE='raw')
            validation=BreastDCEDataset(cfg['VAL_CSV'],root,config=cfg)
            state=torch.get_rng_state().clone();a=validation[0]['image'];b=validation[0]['image']
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
            training=BreastDCEDataset(cfg['TRAIN_CSV'],root,config=cfg,augment=True)
            transformed=training[0]['image']
            self.assertFalse(torch.equal(a,transformed))
            torch.testing.assert_close(transformed[0],transformed[1],atol=0,rtol=0)
            torch.testing.assert_close(transformed[0],transformed[2],atol=0,rtol=0)

    def test_augmented_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=phase4_tests.Phase4Tests().fixture(root)
            for phase,offset in [('PRE',0),('EARLY',15),('LATE',30)]:
                image=((np.arange(256).reshape(16,16)+offset)%256).astype(np.uint8)
                Image.fromarray(image).save(root/f'{phase}.png')
            cfg.update(USE_AUGMENTATION=True,AUGMENTATION_MODE='gentle_affine',AUGMENT_PROBABILITY=1)
            whole,_=train(cfg);first,_=train(dict(cfg,EPOCHS=1))
            continued,_=train(cfg,resume_from=first/'epoch_001.pt')
            a=torch.load(whole/'epoch_002.pt',weights_only=True)
            b=torch.load(continued/'epoch_002.pt',weights_only=True)
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],atol=0,rtol=0)
            torch.testing.assert_close(a['rng_state']['torch'],b['rng_state']['torch'],atol=0,rtol=0)

    def test_runner_rejects_ambiguous_restart(self):
        with patch('sys.argv',['batch_train','--force-new','--resume-interrupted']):
            with self.assertRaises(SystemExit) as context:batch_main()
            self.assertEqual(context.exception.code,2)


if __name__=='__main__':unittest.main()
