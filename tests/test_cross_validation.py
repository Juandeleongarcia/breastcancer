"""CV oficial: entidades, train-only, perfiles, reanudación y OOF sin test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.configuration import load_config, canonical_csv_hash, apply_statistics
from src.cv_data import audit_samples, fold_frames, _quantile, fit_fold_statistics
from src.cross_validation import (prepare_campaign, checked_campaign, fit_job,
                                  run_campaign, aggregate_oof, summarize_campaign, save_json, portable)
from src.checkpointing import load_resume
from src.metrics import binary_metrics


class CrossValidationTests(unittest.TestCase):
    def fixture(self,root):
        rows=[];patients=[]
        grid=np.arange(256).reshape(16,16)
        for fold in [-1,0,1,2,3,4]:
            split='test' if fold==-1 else 'train'
            for label in [0,1]:
                pid=f'P{fold+1}_{label}'
                folder=root/'dataset'/split/pid;folder.mkdir(parents=True)
                patients.append(dict(pid=pid,pCR=label,split=split,test=int(split=='test')))
                for z in [0,1]:
                    pre=(grid+fold*11+label*17+z*3)%160
                    early=pre+10+grid%9+label*7;late=pre+5+grid%5
                    paths={}
                    for phase,array in [('PRE',pre),('EARLY',early),('LATE',late)]:
                        name=f'{pid}_z{z:03d}_{phase}.png'
                        Image.fromarray(array.astype(np.uint8)).save(folder/name)
                        paths['path_'+{'PRE':'pre','EARLY':'early','LATE':'late'}[phase]]=(folder/name).relative_to(root).as_posix()
                    rows.append(dict(sample_id=f'{pid}_z{z:03d}',patient_id=pid,split=split,fold=fold,pCR=label,**paths))
        frame=pd.DataFrame(rows);frame.to_csv(root/'samples.csv',index=False)
        pd.DataFrame(patients).to_csv(root/'patients.csv',index=False)
        protocol=dict(samples_csv=str(root/'samples.csv'),patients_csv=str(root/'patients.csv'),
                      samples_sha256=canonical_csv_hash(root/'samples.csv'),data_dir=str(root))
        template=load_config('config/phase9/SPATIAL_SMALL.json')
        template.update(IMAGE_SIZE=16,BATCH_SIZE=2,EPOCHS=2,TORCH_NUM_THREADS=2)
        return frame,protocol,template

    def test_real_official_counts_and_reserved_test(self):
        protocol=load_config('config/phase9/protocol.json')
        frame,report=audit_samples(protocol['samples_csv'],protocol['patients_csv'],protocol['samples_sha256'])
        self.assertEqual(report['train_patients'],1097)
        self.assertEqual(report['reserved_test_patients'],176)
        self.assertEqual([x['val_slices'] for x in report['folds']],[2186,2190,2192,2192,2185])
        self.assertEqual([x['val_patient_pcr'] for x in report['folds']],[64,62,59,75,62])
        seen=set()
        for fold in range(5):
            tr,va=fold_frames(frame,fold)
            self.assertFalse(set(tr.patient_id)&set(va.patient_id))
            self.assertTrue(tr.split.eq('train').all() and va.split.eq('train').all())
            self.assertFalse(seen&set(va.patient_id));seen.update(va.patient_id)
        self.assertEqual(len(seen),1097)

    def test_patient_may_not_change_fold_and_test_may_not_get_a_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,protocol,_=self.fixture(root)
            bad=frame.copy();bad.loc[bad.index[bad.patient_id.eq('P1_0')][0],'fold']=2
            bad.to_csv(root/'bad.csv',index=False)
            with self.assertRaisesRegex(ValueError,'paciente'):
                audit_samples(root/'bad.csv')
            bad=frame.copy();bad.loc[bad.split.eq('test'),'fold']=0
            bad.to_csv(root/'bad.csv',index=False)
            with self.assertRaisesRegex(ValueError,'test'):
                audit_samples(root/'bad.csv')

    def test_histogram_quantiles_equal_full_linear_quantiles(self):
        values=np.array([-255,-10,-1,0,1,2,10,255])
        hist=np.bincount(values+255,minlength=511)
        for q in [.01,.5,.99]:
            self.assertAlmostEqual(_quantile(hist,q),float(np.quantile(values/255,q)),places=13)

    def test_statistics_read_only_authorized_train_and_match_pixel_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template)
            tr,va=fold_frames(frame,0)
            original_open=Image.open
            def guarded(path,*args,**kwargs):
                self.assertFalse(any(str(pid) in str(path) for pid in va.patient_id.unique()))
                self.assertNotIn('/test/',Path(path).as_posix())
                return original_open(path,*args,**kwargs)
            with patch('src.cv_data.Image.open',side_effect=guarded):
                cfg=fit_job(campaign,0)
            arrays=[]
            for r in tr.itertuples():
                phases=[np.asarray(original_open(root/getattr(r,c)),dtype=np.float64)/255 for c in ['path_pre','path_early','path_late']]
                arrays.append(np.stack([phases[0],phases[1]-phases[0],phases[2]-phases[0]]))
            x=np.stack(arrays)
            for c in range(3):
                lo,hi=np.quantile(x[:,c],[.01,.99])
                clipped=np.clip(x[:,c],lo,hi)
                self.assertAlmostEqual(cfg['CHANNEL_P01'][c],lo,places=12)
                self.assertAlmostEqual(cfg['CHANNEL_P99'][c],hi,places=12)
                self.assertAlmostEqual(cfg['CHANNEL_MEANS'][c],clipped.mean(),places=12)
                self.assertAlmostEqual(cfg['CHANNEL_STDS'][c],clipped.std(),places=12)
            self.assertEqual(cfg['STATS_PROVENANCE'],'verified_train_fit')

    def test_profiles_cannot_be_reused_across_fold_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template)
            a=fit_job(campaign,0);b=fit_job(campaign,1)
            with self.assertRaisesRegex(ValueError,'estadísticas'):
                apply_statistics(dict(b),a['STATS_PATH'])
            (campaign/'fold_0/train.csv').write_bytes((campaign/'fold_0/train.csv').read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'alterado'):
                checked_campaign(campaign)

    def test_oof_requires_unique_full_coverage_correct_fold_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,_,_=self.fixture(Path(tmp))
            patients=frame[frame.split.eq('train')][['patient_id','pCR','fold']].drop_duplicates()
            pred=pd.DataFrame(dict(patient_id=patients.patient_id,label=patients.pCR,
                                   probability=np.where(patients.pCR,.9,.1),cv_fold=patients.fold))
            self.assertEqual(aggregate_oof(frame,pred)['auc'],1.)
            with self.assertRaises(ValueError):aggregate_oof(frame,pred.iloc[:-1])
            with self.assertRaises(ValueError):aggregate_oof(frame,pd.concat([pred,pred.iloc[:1]]))
            bad=pred.copy();bad.iloc[0,bad.columns.get_loc('cv_fold')]=4
            with self.assertRaises(ValueError):aggregate_oof(frame,bad)

    def test_full_cv_cannot_start_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template)
            with patch('src.experiment.train') as training:
                with self.assertRaisesRegex(RuntimeError,'ROCm'):
                    run_campaign(campaign,[0],device='cpu')
                training.assert_not_called()

    def test_synthetic_smoke_files_balanced_metrics_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template,smoke=True)
            summary=run_campaign(campaign,[0],device='cpu')
            self.assertTrue(summary['diagnostic']);self.assertIsNone(summary['oof_patient_auc'])
            self.assertEqual(summary['failed_folds'],0)
            state=load_config(campaign/'campaign.json')
            self.assertEqual(state['folds']['0']['status'],'complete')
            directory=Path(state['folds']['0']['last_run'])
            if not directory.is_absolute():
                from src.configuration import project_path
                directory=project_path(directory)
            history=pd.read_csv(directory/'history.csv')
            self.assertEqual(len(history),2)
            for part in ['train_eval','val']:
                self.assertIn(part+'_patient_balanced_accuracy',history)
            for name in ['epoch_001.pt','epoch_002.pt','best_epoch.pt','loss_curve.png','loss_comparison.png','accuracy_curve.png','auc_curve.png','roc_curve.png']:
                self.assertTrue((directory/name).is_file(),name)
            checkpoint=torch.load(directory/'epoch_001.pt',weights_only=True)
            metrics=checkpoint['val_metrics']['patient']
            self.assertEqual(metrics['confusion_matrix'],[[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
            self.assertEqual(sum(sum(row) for row in metrics['confusion_matrix']),2)
            cfg=fit_job(campaign,0)
            self.assertEqual(load_resume(directory/'epoch_001.pt',dict(cfg,DEVICE='cpu'))['epoch'],1)
            different=fit_job(campaign,1)
            with self.assertRaisesRegex(ValueError,'incompatible'):
                load_resume(directory/'epoch_001.pt',different)
            before=set((campaign/'fold_0/runs').iterdir())
            run_campaign(campaign,[0],device='cpu',resume=True)
            self.assertEqual(before,set((campaign/'fold_0/runs').iterdir()))

    def test_balanced_accuracy_does_not_reward_majority_only(self):
        result=binary_metrics([0,0,0,1],[.1,.1,.1,.1])
        self.assertEqual(result['accuracy'],.75)
        self.assertEqual(result['balanced_accuracy'],.5)

    def test_summary_mean_sample_std_and_oof_only_after_five_completed_folds(self):
        # Registros fabricados SOLO como fixture temporal: no son resultados de CNN.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template)
            state=load_config(campaign/'campaign.json')
            for fold in range(5):
                directory=campaign/f'fixture_result_{fold}';(directory/'internals').mkdir(parents=True)
                labels=frame[frame.split.eq('train') & frame.fold.eq(fold)][['patient_id','pCR']].drop_duplicates()
                scores=np.where(labels.pCR,.9,.1)
                if fold==4:scores=np.where(labels.pCR,.1,.9)
                metrics=binary_metrics(labels.pCR,scores)
                save_json(directory/'internals/summary.json',dict(best_epoch=1,epochs_completed=1,
                    diagnostic_partial=False,validation=dict(loss=.7,patient=metrics,slice=metrics)))
                pd.DataFrame(dict(epoch=[1],train_eval_loss=[.5])).to_csv(directory/'history.csv',index=False)
                pd.DataFrame(dict(patient_id=labels.patient_id,label=labels.pCR,probability=scores)).to_csv(directory/'internals/predictions_patient.csv',index=False)
                state['folds'][str(fold)].update(status='complete',last_run=portable(directory))
            save_json(campaign/'campaign.json',state)
            result=summarize_campaign(campaign)
            self.assertAlmostEqual(result['auc_mean'],.8)
            self.assertAlmostEqual(result['auc_std_sample'],np.sqrt(.2))
            self.assertAlmostEqual(result['oof_patient_auc'],.8)
            self.assertEqual(result['oof_patient']['n'],10)
            state['folds']['4']['status']='prepared';save_json(campaign/'campaign.json',state)
            partial=summarize_campaign(campaign)
            self.assertTrue(partial['provisional']);self.assertIsNone(partial['oof_patient_auc'])

    def test_interruption_requires_explicit_resume_and_preserves_source_run(self):
        from src.experiment import run_epoch as actual_epoch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,protocol,template=self.fixture(root)
            campaign=prepare_campaign(output_root=root/'outputs',protocol=protocol,template=template,smoke=True)
            calls=0
            def interrupt_after_first(*args,**kwargs):
                nonlocal calls
                optimizer=args[4] if len(args)>4 else kwargs.get('optimizer')
                if optimizer is not None:
                    calls+=1
                    if calls==2:raise KeyboardInterrupt()
                return actual_epoch(*args,**kwargs)
            with patch('src.experiment.run_epoch',side_effect=interrupt_after_first):
                run_campaign(campaign,[0],device='cpu')
            state=load_config(campaign/'campaign.json')
            self.assertEqual(state['folds']['0']['status'],'interrupted')
            source=Path(state['folds']['0']['last_run'])
            if not source.is_absolute():
                from src.configuration import project_path
                source=project_path(source)
            before=(source/'epoch_001.pt').read_bytes()
            with patch('src.experiment.train') as training:
                failed=run_campaign(campaign,[0],device='cpu',resume=False)
                training.assert_not_called();self.assertEqual(failed['failed_folds'],1)
            result=run_campaign(campaign,[0],device='cpu',resume=True)
            self.assertEqual(result['failed_folds'],0)
            state=load_config(campaign/'campaign.json')
            self.assertEqual(state['folds']['0']['status'],'complete')
            self.assertNotEqual(state['folds']['0']['last_run'],portable(source))
            self.assertEqual(before,(source/'epoch_001.pt').read_bytes())


if __name__=='__main__':
    unittest.main()
