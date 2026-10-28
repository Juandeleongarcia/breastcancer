"""F10: orquestación simulada en temporales; ninguna campaña real ni GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd

from src import phase10 as p
from src.configuration import load_config


class Phase10Tests(unittest.TestCase):
    def test_reuses_real_f8_hyperparameters_with_fold_fitting_only(self):
        configs=p.templates()
        for name,lr,drop in [(p.SMALL,.0005,.4),(p.DROPOUT,.0005,.5),(p.LOW_LR,.0001,.4)]:
            cfg=configs[name]
            self.assertEqual(cfg['LEARNING_RATE'],lr);self.assertEqual(cfg['DROPOUT'],drop)
            self.assertEqual(cfg['OPTIMIZER'],'adamw');self.assertEqual(cfg['WEIGHT_DECAY'],.0005)
            self.assertEqual(cfg['BATCH_SIZE'],16);self.assertEqual(cfg['SEED'],42)
            self.assertEqual(cfg['EPOCHS'],20);self.assertEqual(cfg['PATIENCE'],8)
            self.assertFalse(cfg['USE_AUGMENTATION']);self.assertFalse(cfg['USE_WEIGHTED_LOSS'])
            self.assertIsNone(cfg['STATS_PATH']);self.assertEqual(cfg['STATS_PROVENANCE'],'requires_fit')
            self.assertEqual(cfg['CONV_CHANNELS'],[16,32,64])

    def test_proc_detects_trainers_ignores_self_zombies_and_unrelated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for pid,module,state in [(1,'src.cross_validation','R'),(2,'unrelated','R'),(3,'src.train','Z'),(99,'src.train','R')]:
                directory=root/str(pid);directory.mkdir()
                (directory/'cmdline').write_bytes(f'python\0-u\0-m\0{module}\0run\0'.encode())
                (directory/'stat').write_text(f'{pid} (python) {state} 0 0')
            self.assertEqual(p.active_trainers(root,own=99),[1])
            with patch.object(Path,'read_bytes',side_effect=PermissionError()):
                with self.assertRaisesRegex(RuntimeError,'verificar'):
                    p.active_trainers(root,own=99)

    def test_corrupt_checkpoint_is_preserved_and_previous_valid_one_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'RUN_fake';(run/'internals').mkdir(parents=True)
            first=run/'epoch_001.pt';first.write_bytes(b'fixture1')
            bad=run/'epoch_003.pt';bad.write_bytes(b'corrupt fixture')
            recovery=run/'internals/recovery_epoch_start.pt';recovery.write_bytes(b'fixture2')
            def load(filename,cfg):
                if filename==bad:raise RuntimeError('corrupt')
                return dict(epoch=2 if filename==recovery else 1)
            with patch('src.checkpointing.load_resume',side_effect=load),patch('src.resume_validation.validate_resume_environment'):
                chosen,rejected=p.checkpoint_candidates(dict(RESULTS_DIR=str(root)))
            self.assertEqual(chosen,recovery);self.assertEqual(len(rejected),1)
            self.assertEqual(bad.read_bytes(),b'corrupt fixture')

    def fixture(self,root,failed=False):
        state=dict(folds={str(f):dict(status='complete' if f<2 else 'error' if failed else 'ready',
                                     last_run=str(root/f'old_{f}'),config_file='config.json',error='original GPU error' if f==2 else None)
                          for f in range(5)})
        p.save(root/'campaign.json',state)
        def inspect(_):
            s=p.read(root/'campaign.json');rows=[]
            for f in range(5):
                e=s['folds'][str(f)]
                rows.append(dict(fold=f,status='COMPLETO' if e['status']=='complete' else 'RECUPERABLE',run=e['last_run']))
            return root,s,rows
        def fit(_,fold):
            return dict(CV_FOLD=fold,EPOCHS=20,RESULTS_DIR=str(root/f'fold_{fold}/runs'))
        return state,inspect,fit

    def test_completed_folds_skipped_failed_fold_resumes_then_three_four_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);initial,inspect,fit=self.fixture(root,failed=True)
            launched=[];current={}
            def checkpoint(cfg):
                return (root/'recovery_2.pt' if cfg['CV_FOLD']==2 else None),[]
            def execute(args):
                launched.append(args)
                fold=2 if '--resume' in args else int(Path(args[args.index('--config')+1]).parent.name.split('_')[1])
                current['fold']=fold;return 0
            def complete(cfg):return dict(run=str(root/f'new_{cfg["CV_FOLD"]}'))
            with patch('src.phase10.inspect_campaign',side_effect=inspect),patch('src.phase10.active_trainers',return_value=[]),patch(
                    'src.cross_validation.fit_job',side_effect=fit),patch('src.cross_validation.checked_campaign',side_effect=lambda _: (root,p.read(root/'campaign.json'),None)),patch(
                    'src.phase10.checkpoint_candidates',side_effect=checkpoint),patch('src.phase10.complete_details',side_effect=complete),patch(
                    'src.cross_validation.summarize_campaign',return_value={}):
                p.run_manual(root,execute=execute)
                p.run_manual(root,execute=execute)
            self.assertEqual(len(launched),3)
            self.assertIn('--resume',launched[0]);self.assertIn('--config',launched[1]);self.assertIn('--config',launched[2])
            final=p.read(root/'campaign.json')
            self.assertEqual(initial['folds']['0'],final['folds']['0'])
            self.assertEqual(initial['folds']['1'],final['folds']['1'])
            self.assertTrue(all(e['status']=='complete' for e in final['folds'].values()))

    def test_gpu_error_stops_after_one_child_without_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,inspect,fit=self.fixture(root)
            with patch('src.phase10.inspect_campaign',side_effect=inspect),patch('src.phase10.active_trainers',return_value=[]),patch(
                    'src.cross_validation.fit_job',side_effect=fit),patch('src.cross_validation.checked_campaign',side_effect=lambda _: (root,p.read(root/'campaign.json'),None)),patch(
                    'src.phase10.checkpoint_candidates',return_value=(root/'recovery.pt',[])),patch('src.phase10.complete_details',return_value=None):
                execute=Mock(return_value=1);execute.stop=False
                with self.assertRaisesRegex(RuntimeError,'no se reintenta'):
                    p.run_manual(root,[2,3,4],execute=execute)
                self.assertEqual(execute.call_count,1)
            state=p.read(root/'campaign.json')
            self.assertEqual(state['folds']['2']['status'],'error')
            self.assertEqual(state['folds']['3']['status'],'ready')

    def test_restart_requires_explicit_flag_and_keeps_failed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,inspect,fit=self.fixture(root)
            old=root/'fold_2/runs/RUN_failed';old.mkdir(parents=True);(old/'failed.pt').write_bytes(b'preserve')
            with patch('src.phase10.inspect_campaign',side_effect=inspect),patch('src.phase10.active_trainers',return_value=[]),patch(
                    'src.cross_validation.fit_job',side_effect=fit),patch('src.cross_validation.checked_campaign',side_effect=lambda _: (root,p.read(root/'campaign.json'),None)),patch(
                    'src.phase10.checkpoint_candidates',return_value=(None,[])),patch('src.phase10.complete_details',return_value=dict(run=str(root/'new'))),patch(
                    'src.cross_validation.summarize_campaign',return_value={}):
                execute=Mock(return_value=0);execute.stop=False
                with self.assertRaisesRegex(ValueError,'restart-fold'):
                    p.run_manual(root,[2],execute=execute)
                execute.assert_not_called()
                p.run_manual(root,[2],restart_folds=[2],execute=execute)
                self.assertIn('--config',execute.call_args.args[0])
                state=p.read(root/'campaign.json');state['folds']['2']['status']='error';p.save(root/'campaign.json',state)
                with patch('src.phase10.checkpoint_candidates',return_value=(root/'valid_but_unusable.pt',[])):
                    p.run_manual(root,[2],restart_folds=[2],execute=execute)
                self.assertNotIn('--resume',execute.call_args.args[0])
            self.assertEqual((old/'failed.pt').read_bytes(),b'preserve')
            self.assertEqual(p.read(root/'campaign.json')['folds']['2']['phase10_attempts'][0]['mode'],'restart_preserving_old_runs')

    def test_variants_reuse_registry_and_adopt_creation_before_registry_save(self):
        from test_cross_validation import CrossValidationTests
        from src.cross_validation import prepare_campaign as original
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,protocol,_=CrossValidationTests().fixture(root)
            def prepare(model,output_root,template):
                return original(model,output_root,protocol=protocol,template=template)
            with patch('src.cross_validation.prepare_campaign',side_effect=prepare) as factory:
                first=p.variant_campaign(p.DROPOUT,root/'state')
                second=p.variant_campaign(p.DROPOUT,root/'state')
                self.assertEqual(first,second);self.assertEqual(factory.call_count,1)
                self.assertEqual(load_config(first/'campaign.json')['model'],p.DROPOUT)
            actual_save=p.save
            def fail_registry(filename,value):
                if Path(filename).name=='campaigns.json':raise RuntimeError('simulated interruption')
                actual_save(filename,value)
            with patch('src.cross_validation.prepare_campaign',side_effect=prepare),patch('src.phase10.save',side_effect=fail_registry):
                with self.assertRaises(RuntimeError):p.variant_campaign(p.LOW_LR,root/'state')
            with patch('src.cross_validation.prepare_campaign',side_effect=AssertionError('duplicate')):
                recovered=p.variant_campaign(p.LOW_LR,root/'state')
            self.assertEqual(len(list((root/'state/campaigns'/p.LOW_LR).glob('CV9_*'))),1)
            self.assertEqual(load_config(recovered/'template.json')['LEARNING_RATE'],.0001)

    def test_incomplete_comparison_does_not_invent_means_or_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);campaign=root/'small';campaign.mkdir()
            folds=[dict(fold=f,status='COMPLETO' if f<2 else 'PENDIENTE') for f in range(5)]
            with patch('src.phase10.inspect_campaign',return_value=(campaign,{},folds)),patch(
                    'src.cross_validation.summarize_campaign',side_effect=AssertionError('false complete')):
                rows=p.comparison(root/'state',str(campaign))
            self.assertEqual(rows[0]['complete_folds'],2)
            self.assertTrue(all(r['status']=='INCOMPLETA' for r in rows))
            self.assertTrue(all(r.get('oof_auc') is None for r in rows))

    def test_summary_alone_does_not_prove_a_completed_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'internals').mkdir()
            p.save(root/'internals/summary.json',dict(epochs_completed=1,best_epoch=1,stopped_early=True,
                validation_complete=True,diagnostic_partial=False))
            pd.DataFrame(dict(epoch=[1])).to_csv(root/'history.csv',index=False)
            with patch('src.batch_train.completed_run',return_value=root):
                with self.assertRaises(FileNotFoundError):p.complete_details(dict(EPOCHS=20))

    def test_missing_class_campaign_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing=Path(tmp)/'class_campaign'
            with self.assertRaises(FileNotFoundError):p.inspect_campaign(missing)
            self.assertFalse(missing.exists())

    def test_trainer_appearing_during_statistics_prevents_gpu_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,inspect,fit=self.fixture(root)
            with patch('src.phase10.inspect_campaign',side_effect=inspect),patch('src.phase10.active_trainers',side_effect=[[],[1234]]),patch(
                    'src.cross_validation.fit_job',side_effect=fit),patch('src.cross_validation.checked_campaign',side_effect=lambda _: (root,p.read(root/'campaign.json'),None)),patch(
                    'src.phase10.checkpoint_candidates',return_value=(root/'recovery.pt',[])):
                execute=Mock();execute.stop=False
                with self.assertRaisesRegex(RuntimeError,'antes de lanzar'):
                    p.run_manual(root,[2],execute=execute)
                execute.assert_not_called()
            self.assertEqual(p.read(root/'campaign.json')['folds']['2']['status'],'ready')


if __name__=='__main__':unittest.main()
