"""CNN4: pruebas sintéticas en temporales, sin dataset ni campañas de clase."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from zipfile import ZipFile

import pandas as pd
import torch
from torch import nn

from src import phase11 as p
from src.configuration import load_config
from src.phase11_model import (ARCHITECTURE, BLOCK_CHANNELS, DCEFourBlockCNN,
                               TRAINABLE_PARAMETERS)


class Phase11Tests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_exact_blocks_shapes_logits_gradients_and_parameter_count(self):
        model = DCEFourBlockCNN()
        conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertEqual([(m.in_channels, m.out_channels) for m in conv],
                         [(3,8),(8,8),(8,16),(16,16),(16,32),(32,32),(32,64),(64,64)])
        self.assertTrue(all(m.bias is None and m.kernel_size == (3,3) and m.padding == (1,1) for m in conv))
        self.assertEqual(len([m for m in model.modules() if isinstance(m, nn.MaxPool2d)]), 3)
        self.assertFalse(any(isinstance(m, nn.Sigmoid) for m in model.modules()))
        self.assertEqual(sum(m.weight.numel() for m in conv), 73368)
        bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertEqual([m.num_features for m in bn], [8,8,16,16,32,32,64,64])
        self.assertEqual(sum(v.numel() for m in bn for v in m.parameters()), 480)
        self.assertEqual(sum(v.numel() for v in model.classifier.parameters()), 65)
        self.assertEqual(sum(v.numel() for v in model.parameters() if v.requires_grad), TRAINABLE_PARAMETERS)
        shapes = []
        handles = [block.register_forward_hook(lambda module, args, output: shapes.append(tuple(output.shape)))
                   for block in model.blocks]
        logits = model(torch.randn(2,3,256,256))
        for handle in handles: handle.remove()
        self.assertEqual(shapes, [(2,8,128,128),(2,16,64,64),(2,32,32,32),(2,64,32,32)])
        self.assertEqual(tuple(logits.shape), (2,1))
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([[0.],[1.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
        self.assertEqual(model.classifier[2].p, .30)
        with torch.no_grad():
            model.classifier[-1].weight.zero_()
            model.classifier[-1].bias.fill_(2.)
            self.assertTrue(torch.equal(model.eval()(torch.randn(2,3,256,256)), torch.full((2,1),2.)))

    def test_seed_reproducibility_and_real_configuration(self):
        from src.experiment import set_seed
        first = p.template()
        second = p.template()
        self.assertEqual(first, second)
        wanted = dict(OPTIMIZER='adamw',LEARNING_RATE=.001,WEIGHT_DECAY=.0001,BATCH_SIZE=16,
                      EPOCHS=10,PATIENCE=8,SEED=42,DROPOUT=.30,SCHEDULER_PATIENCE=3,
                      SCHEDULER_FACTOR=.5,USE_WEIGHTED_LOSS=False,EVALUATE_TRAIN=True)
        self.assertTrue(all(first[k] == v for k,v in wanted.items()))
        outputs = []
        for _ in range(2):
            set_seed(42)
            model = DCEFourBlockCNN(first)
            outputs.append(model(torch.randn(2,3,32,32)).detach())
        self.assertTrue(torch.equal(*outputs))
        changed = deepcopy(first);changed['PHASE11_SOURCES']['src/phase11_model.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'fuentes'):
            p.verify_config(changed)

    def test_runtime_is_reversible_and_legacy_factory_contract_stay_equivalent(self):
        from src import checkpointing, experiment, evaluate
        from src.cross_validation import implementation
        original = (experiment.BreastCancerCNN, checkpointing.contract, experiment.contract,
                    experiment.create_run_directory, experiment.validate, evaluate.BreastCancerCNN)
        frozen = implementation()
        legacy = load_config('config/phase9/SPATIAL_SMALL.json')
        before = checkpointing.contract(legacy)
        cfg = p.template()
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with p.runtime():
                self.assertEqual(checkpointing.contract(legacy), before)
                self.assertIsInstance(experiment.BreastCancerCNN(legacy), original[0])
                self.assertIsInstance(experiment.BreastCancerCNN(cfg), DCEFourBlockCNN)
                identity = checkpointing.contract(cfg)
                self.assertEqual(identity['architecture']['blocks'], BLOCK_CHANNELS)
                self.assertTrue(set(p.SOURCES) <= identity['source_sha256'].keys())
                raise RuntimeError('fixture')
        self.assertEqual(original, (experiment.BreastCancerCNN, checkpointing.contract, experiment.contract,
                         experiment.create_run_directory, experiment.validate, evaluate.BreastCancerCNN))
        self.assertEqual(implementation(), frozen)
        self.assertEqual(checkpointing.contract(legacy), before)

    def fixture(self, root):
        from test_cross_validation import CrossValidationTests
        return CrossValidationTests().fixture(root)

    def test_persistent_campaign_official_folds_and_incompatible_campaign_rejected(self):
        from src.cross_validation import prepare_campaign
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, protocol, _ = self.fixture(root)
            campaign = p.campaign_for(root/'state', create=True, protocol=protocol)
            self.assertEqual(campaign, p.campaign_for(root/'state', create=True))
            self.assertEqual(len(list((root/'state/campaigns').glob('CV9_*'))), 1)
            _, state, _ = p.checked(campaign)
            self.assertEqual(state['model'], ARCHITECTURE)
            for fold in range(5):
                tr = pd.read_csv(campaign/f'fold_{fold}/train.csv')
                va = pd.read_csv(campaign/f'fold_{fold}/validation.csv')
                self.assertTrue(tr.split.eq('train').all() and va.split.eq('train').all())
                self.assertFalse(tr.fold.eq(fold).any());self.assertTrue(va.fold.eq(fold).all())
                self.assertFalse(set(tr.patient_id) & set(va.patient_id))
            old = prepare_campaign(output_root=root/'old', protocol=protocol)
            original_bytes = (old/'campaign.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'ajena'):
                p.campaign_for(root/'other', campaign=old, create=True)
            self.assertEqual((old/'campaign.json').read_bytes(), original_bytes)
            state['phase11_implementation']['src/phase11.py'] = 'tampered'
            p.phase10.save(campaign/'campaign.json', state)
            with self.assertRaisesRegex(ValueError, 'incompatibles'):
                p.checked(campaign)

    def test_creation_interrupted_before_registry_adopts_same_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);_, protocol, _ = self.fixture(root)
            original = p.phase10.save
            def fail(filename, value):
                if Path(filename).name == 'route.json': raise RuntimeError('registry fixture')
                original(filename, value)
            with patch('src.phase10.save', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'registry'):
                    p.campaign_for(root/'state', create=True, protocol=protocol)
            with patch('src.cross_validation.prepare_campaign', side_effect=AssertionError('duplicate')):
                campaign = p.campaign_for(root/'state', create=True)
            self.assertEqual(len(list((root/'state/campaigns').glob('CV9_*'))), 1)
            self.assertEqual(campaign, p.campaign_for(root/'state'))

    def test_cpu_clinical_training_is_rejected_without_creating_outputs(self):
        cfg = p.template();cfg['DEVICE'] = 'cpu'
        with patch('src.experiment.train') as train:
            with self.assertRaisesRegex(RuntimeError, 'ROCm'):
                p.train_job(cfg)
            train.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)/'absent'
            with self.assertRaises(FileNotFoundError):p.campaign_for(missing)
            self.assertFalse(missing.exists())

    def test_child_uses_phase11_worker_and_cli_accepts_progressive_folds(self):
        with patch.object(p.phase10.Executor, '__call__', return_value=0) as execute:
            p.Executor()(['python','-u','-m','src.train','--config','fixture.json','--device','cuda'])
        self.assertEqual(execute.call_args.args[0],
                         ['python','-u','-m','src.phase11','train','--config','fixture.json','--device','cuda'])
        for folds in ([0], [1,2,3,4]):
            args = p.parser().parse_args(['run','--folds',*map(str,folds)])
            self.assertEqual(args.folds, folds)
        self.assertIsNone(p.parser().parse_args(['run']).folds)
        self.assertEqual(p.parser().parse_args(['train','--resume','fixture.pt','--epochs','10']).epochs, 10)

    def test_nonfinite_auc_never_reaches_scheduler(self):
        from src import experiment
        with patch.object(experiment, 'validate', return_value={'patient':{'auc':float('nan')}}):
            with p.runtime():
                with self.assertRaisesRegex(ValueError, 'no avanzar scheduler'):
                    experiment.validate(None,None,None,None,p.template())

    def test_synthetic_checkpoints_curves_resume_optimizer_scheduler_and_evaluation(self):
        from src.cross_validation import fit_job
        from src import checkpointing
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);_, protocol, _ = self.fixture(root)
            cfg = p.template()
            cfg.update(IMAGE_SIZE=16,BATCH_SIZE=2,EPOCHS=2,TORCH_NUM_THREADS=2,SMOKE_TEST=True)
            campaign = p.campaign_for(root/'state', create=True, protocol=protocol, config=cfg)
            job = dict(fit_job(campaign,0), DEVICE='cpu')
            target, summary = p.train_job(job)
            self.assertTrue(summary['diagnostic_partial'])
            self.assertEqual(summary['trainable_parameters'], 73913)
            self.assertEqual(summary['convolutions_per_block'], [1,2,2,3])
            for name in ['config.py','history.csv','epoch_001.pt','epoch_002.pt','best_epoch.pt',
                         'loss_curve.png','accuracy_curve.png','auc_curve.png','roc_curve.png','loss_comparison.png']:
                self.assertTrue((target/name).is_file(), name)
            history = pd.read_csv(target/'history.csv')
            for column in ['train_eval_loss','val_loss','train_eval_patient_accuracy','val_patient_auc']:
                self.assertIn(column, history)
            checkpoint = target/'epoch_001.pt'
            original = checkpoint.read_bytes()
            with p.runtime():
                payload = checkpointing.load_resume(checkpoint, job)
                self.assertTrue(payload['optimizer_state_dict']['state'])
                self.assertIsNotNone(payload['scheduler_state_dict'])
                self.assertEqual(payload['training_contract']['architecture']['id'], ARCHITECTURE)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                checkpointing.load_resume(checkpoint, job)
            resumed, _ = p.train_job(job, checkpoint)
            self.assertNotEqual(resumed, target)
            self.assertEqual(checkpoint.read_bytes(), original)
            first = torch.load(target/'epoch_002.pt', weights_only=True)
            second = torch.load(resumed/'epoch_002.pt', weights_only=True)
            self.assertTrue(all(torch.equal(first['model_state_dict'][k], second['model_state_dict'][k])
                                for k in first['model_state_dict']))
            self.assertEqual(first['optimizer_state_dict']['param_groups'], second['optimizer_state_dict']['param_groups'])
            self.assertEqual(first['scheduler_state_dict'], second['scheduler_state_dict'])
            result = p.evaluate_job(resumed/'best_epoch.pt')
            self.assertEqual(result['patient']['n'], 2)
            self.assertEqual(len(result['patient']['confusion_matrix']), 2)
            with ZipFile(resumed/'internals/sources.zip') as archive:
                self.assertTrue(set(p.SOURCES) <= set(archive.namelist()))


if __name__ == '__main__':
    unittest.main()
