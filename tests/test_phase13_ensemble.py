"""Ensemble: solo cohortes ficticias temporales; prohíbe backward/optimizer/fit."""
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

import test_phase13 as fixtures
from src import phase13 as p
from src import phase13_ensemble as ensemble
from src.configuration import load_config
from src.phase13_clinical import ClinicalTransform
from src.phase13_engine import inference_contract
from src.phase13_inference import ImageTransform
from src.phase13_model import SpatialSmallMultimodal
from src.metrics import binary_metrics


def hashes(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob('*') if path.is_file()}


class Phase13EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.threads=torch.get_num_threads();torch.set_num_threads(2)
        self.backward=patch('torch.Tensor.backward',side_effect=AssertionError('Ensemble no entrena'))
        self.steps=patch('torch.optim.Optimizer.step',side_effect=AssertionError('Ensemble no actualiza pesos'))
        self.backward.start();self.steps.start()

    def tearDown(self):
        self.steps.stop();self.backward.stop();torch.set_num_threads(self.threads);self.temp.cleanup()

    def fixture(self):
        campaign,protocol,cfg=fixtures.Phase13Tests.fixture(self,256,epochs=1,smoke=False)
        self.campaign=campaign;self.protocol=protocol;self.output=self.root/'external_results'
        for fold in range(5):
            job,clinical=p.fit_job(campaign,fold)
            root,state,base,dev,patients,external=p.checked(campaign)
            folder=campaign/f'fold_{fold}'
            run=folder/'runs'/f'RUN_SYNTHETIC_ONLY_{fold}';details=run/'internals';details.mkdir(parents=True)
            model=SpatialSmallMultimodal(job,len(clinical['columns']))
            with torch.no_grad():
                for parameter in model.parameters():parameter.zero_()
                model.classifier[1].weight[0,64]=.2*(fold+1)
                model.classifier[1].bias.fill_((fold-2)*.25)
            validation=p.partition(dev,fold)[1]
            ids=validation.patient_id.unique().tolist()
            values=ClinicalTransform(clinical).transform(patients,ids)
            probabilities=torch.sigmoid(torch.from_numpy(values[:,0])*.2*(fold+1)+(fold-2)*.25).tolist()
            labels=validation.groupby('patient_id').pCR.first().to_dict()
            pred=pd.DataFrame([dict(patient_id=pid,label=labels[pid],probability=prob,n_slices=2)
                               for pid,prob in zip(ids,probabilities)])
            metric=binary_metrics(pred.label,pred.probability)
            row=dict(epoch=1,learning_rate=job['LEARNING_RATE'],train_eval_loss=.7,val_loss=.7,
                     val_patient_auc=metric['auc'],val_slice_auc=metric['auc'])
            pd.DataFrame([row]).to_csv(run/'history.csv',index=False)
            pred.to_csv(details/'predictions_patient.csv',index=False)
            p.write_json(details/'summary.json',dict(status='complete',SMOKE_TEST=False,diagnostic_partial=False,
                validation_complete=True,epochs_completed=1,best_epoch=1,stopped_early=False))
            contract=p.training_contract(state,job,clinical,job['TRAIN_CSV'],job['VAL_CSV'],patients)
            payload=dict(schema_version=13,architecture=ensemble.ARCHITECTURE,epoch=1,best_epoch=1,
                best_auc=metric['auc'],epoch_incomplete=False,config=job,clinical_transform=clinical,
                training_contract=contract,inference_contract=inference_contract(job,clinical),
                history=[row],model_state_dict=model.state_dict())
            torch.save(payload,run/'best_epoch.pt')
            state['folds'][str(fold)].update(status='complete',last_run=run.relative_to(campaign).as_posix())
            p.write_json(campaign/'campaign.json',state)
        return campaign

    def test_average_is_patient_aligned_equal_probability_mean_not_vote(self):
        samples=pd.DataFrame(dict(patient_id=['a','a','b'],sample_id=['a0','a1','b0']))
        predictions={}
        for fold,prob in enumerate([.49,.49,.49,.99,.99]):
            predictions[fold]=pd.DataFrame(dict(patient_id=['b','a'],probability=[1-prob,prob],n_slices=[1,2]))
        result=ensemble.average_patients(predictions,samples).set_index('patient_id')
        self.assertAlmostEqual(result.loc['a','probability'],.69)
        self.assertAlmostEqual(result.loc['b','probability'],.31)
        with self.assertRaises(ValueError):ensemble.average_patients({k:v for k,v in predictions.items() if k!=4},samples)
        bad=dict(predictions);bad[4]=predictions[4].iloc[:1]
        with self.assertRaises(ValueError):ensemble.average_patients(bad,samples)
        bad=dict(predictions);bad[4]=pd.concat([predictions[4],predictions[4].iloc[:1]])
        with self.assertRaises(ValueError):ensemble.average_patients(bad,samples)

    def test_preflight_reads_no_external_images_and_keeps_frozen_sources(self):
        campaign=self.fixture();before=hashes(campaign);sources=p.implementation()
        self.assertNotIn('src/phase13_ensemble.py',sources)
        with patch('PIL.Image.open',side_effect=AssertionError('Preflight no abre imágenes')):
            _,state,external,members,device=ensemble.preflight(campaign,'cpu',_expected_test_patients=2)
        self.assertEqual(len(members),5);self.assertEqual(len(external),2);self.assertEqual(str(device),'cpu')
        self.assertGreater(len({tuple(m.config['CHANNEL_MEANS']) for m in members}),1)
        self.assertGreater(len({m.clinical['age_mean'] for m in members}),1)
        self.assertEqual(sources,p.implementation());self.assertEqual(before,hashes(campaign))

    def test_end_to_end_uses_own_transforms_and_matches_manual_five_model_oracle(self):
        campaign=self.fixture();before=hashes(campaign);original_data=hashes(Path(self.protocol['data_dir']))
        _,_,_,members,_=ensemble.preflight(campaign,'cpu',_expected_test_patients=2)
        raw=pd.read_csv(self.protocol['samples_csv']);test=raw[raw.fold.eq(-1)]
        patients=p.clinical_table(self.protocol['patients_csv'],test.patient_id.unique())
        ids=sorted(test.patient_id.unique());oracle={}
        for member in members:
            age_z=ClinicalTransform(member.clinical).transform(patients,ids)[:,0]
            oracle[member.fold]=torch.sigmoid(torch.from_numpy(age_z)*.2*(member.fold+1)+(member.fold-2)*.25).numpy()
        real_predict=ensemble.predict_member;real_item=ensemble.ExternalDataset.__getitem__
        active={};seen=[]
        def predict(member,*args):
            active['fold']=member.fold
            return real_predict(member,*args)
        def item(dataset,index):
            value=real_item(dataset,index)
            self.assertNotIn('pCR',dataset.frame);self.assertNotIn('label',value)
            if index==0:
                row=dataset.frame.iloc[index]
                phases=np.stack([np.asarray(Image.open(Path(self.protocol['data_dir'])/row[key]))
                                 for key in ['path_pre','path_early','path_late']])
                expected=ImageTransform(members[active['fold']].config)(phases)
                self.assertTrue(torch.equal(value['image'],expected));seen.append(active['fold'])
            return value
        with patch('src.phase13_ensemble.predict_member',side_effect=predict), \
             patch.object(ensemble.ExternalDataset,'__getitem__',new=item), \
             patch('src.phase13_clinical.ClinicalTransform.fit',side_effect=AssertionError('No reajustar clínica')), \
             patch('src.phase13.fit_fold_statistics',side_effect=AssertionError('No reajustar imágenes')):
            report,result=ensemble.evaluate_ensemble(campaign,'cpu',True,_output_root=self.output,_expected_test_patients=2)
        self.assertEqual(seen,[0,1,2,3,4])
        saved=pd.read_csv(report/'predictions_patient.csv').sort_values('patient_id')
        for fold in range(5):np.testing.assert_allclose(saved[f'probability_fold_{fold}'],oracle[fold],rtol=1e-6,atol=1e-7)
        expected=np.mean(np.stack([oracle[f] for f in range(5)],axis=1).astype(float),axis=1)
        np.testing.assert_allclose(saved.probability,expected,rtol=1e-6)
        manual=binary_metrics(saved.label,expected,.5)
        for key in ['auc','accuracy','balanced_accuracy','recall','specificity','confusion_matrix']:
            self.assertEqual(result['patient'][key],manual[key])
        self.assertEqual(result['n_models'],5);self.assertEqual(result['n_patients'],2)
        self.assertFalse(result['training_performed'])
        for name in ['metrics.json','metrics.csv','predictions_patient.csv','roc_curve.png',
                     'confusion_matrix_patient.png','evaluation_manifest.json']:
            self.assertTrue((report/name).is_file(),name)
        self.assertEqual(load_config(report/'evaluation_manifest.json')['status'],'complete')
        self.assertEqual(before,hashes(campaign));self.assertEqual(original_data,hashes(Path(self.protocol['data_dir'])))

    def test_confirmation_and_default_176_patient_guard(self):
        with patch('src.phase13_ensemble.preflight',side_effect=AssertionError('Sin autorización no leer datos')):
            with self.assertRaises(ValueError):ensemble.evaluate_ensemble('DO_NOT_READ',confirm=False)
        campaign=self.fixture()
        with self.assertRaisesRegex(ValueError,'completo'):ensemble.preflight(campaign,'cpu')

    def test_incomplete_campaign_rejected_before_external_access(self):
        campaign=self.fixture();state=load_config(campaign/'campaign.json')
        state['folds']['4']['status']='running';p.write_json(campaign/'campaign.json',state)
        with patch('src.phase13_ensemble.external_samples',side_effect=AssertionError('No consultar test')):
            with self.assertRaisesRegex(ValueError,'Fold 4 no completo'):
                ensemble.evaluate_ensemble(campaign,confirm=True,_output_root=self.output,_expected_test_patients=2)
        self.assertFalse(self.output.exists())

    def test_wrong_fold_smoke_epoch_and_clinical_state_are_rejected(self):
        campaign=self.fixture();state=load_config(campaign/'campaign.json')
        checkpoint=campaign/state['folds']['0']['last_run']/'best_epoch.pt';original=checkpoint.read_bytes()
        for kind in ['fold','smoke','epoch','clinical','threshold','nonfinite']:
            with self.subTest(kind=kind):
                checkpoint.write_bytes(original);payload=torch.load(checkpoint,weights_only=True)
                if kind=='fold':payload['config']['CV_FOLD']=1
                if kind=='smoke':payload['config']['SMOKE_TEST']=True
                if kind=='epoch':payload['epoch']=2
                if kind=='clinical':payload['clinical_transform']['age_mean']+=1
                if kind=='threshold':payload['config']['CLASSIFICATION_THRESHOLD']=.2
                if kind=='nonfinite':payload['model_state_dict']['classifier.1.bias'].fill_(float('nan'))
                torch.save(payload,checkpoint)
                with self.assertRaises(ValueError):ensemble.preflight(campaign,'cpu',_expected_test_patients=2)

    def test_single_evaluation_is_reserved_even_if_model_weights_change(self):
        campaign=self.fixture()
        ensemble.evaluate_ensemble(campaign,confirm=True,_output_root=self.output,_expected_test_patients=2)
        state=load_config(campaign/'campaign.json');checkpoint=campaign/state['folds']['0']['last_run']/'best_epoch.pt'
        payload=torch.load(checkpoint,weights_only=True);payload['model_state_dict']['classifier.1.bias']+=.1
        torch.save(payload,checkpoint)
        with patch('src.phase13_ensemble.external_samples',side_effect=AssertionError('No repetir test')):
            with self.assertRaisesRegex(ValueError,'ya tiene una evaluación'):
                ensemble.evaluate_ensemble(campaign,confirm=True,_output_root=self.output,_expected_test_patients=2)

    def test_failure_keeps_reservation_without_partial_metrics(self):
        campaign=self.fixture()
        with patch('src.phase13_ensemble.predict_member',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError):
                ensemble.evaluate_ensemble(campaign,confirm=True,_output_root=self.output,_expected_test_patients=2)
        ledgers=list((self.output/'_evaluations').glob('*.json'))
        self.assertEqual(len(ledgers),1);self.assertEqual(load_config(ledgers[0])['status'],'failed')
        self.assertFalse(list(self.output.glob('ENSEMBLE13_*/metrics.json')))
        with self.assertRaisesRegex(ValueError,'ya tiene una evaluación'):
            ensemble.evaluate_ensemble(campaign,confirm=True,_output_root=self.output,_expected_test_patients=2)

    def test_cli_help_exposes_device_and_only_explicit_test_activation(self):
        with patch('sys.argv',['ensemble','--help']),redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit) as error:ensemble.main()
            self.assertEqual(error.exception.code,0)
        for name in ['--campaign','--device','cuda','--check-only','--confirm-external-test']:
            self.assertIn(name,output.getvalue())
        self.assertNotIn('--threshold',output.getvalue());self.assertNotIn('--folds',output.getvalue())

    def test_member_aggregates_sigmoid_probabilities_of_all_slices_not_logits(self):
        campaign=self.fixture()
        _,_,_,members,device=ensemble.preflight(campaign,'cpu',_expected_test_patients=2)
        frame=pd.read_csv(self.protocol['samples_csv']);samples=frame[frame.fold.eq(-1)].drop(columns='pCR')
        patients=p.clinical_table(self.protocol['patients_csv'],samples.patient_id.unique())
        logits=torch.tensor([-3.,1.,0.,2.])
        def forward(model,images,clinical):
            self.assertFalse(model.training);self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            return logits[:,None]
        with patch.object(SpatialSmallMultimodal,'forward',new=forward):
            result,slices=ensemble.predict_member(members[0],samples,patients,self.protocol['data_dir'],device)
        expected=torch.sigmoid(logits).reshape(2,2).mean(dim=1).numpy()
        np.testing.assert_allclose(result.probability,expected,rtol=1e-7)
        self.assertFalse(np.allclose(result.probability,torch.sigmoid(logits.reshape(2,2).mean(dim=1)).numpy()))
        self.assertEqual(len(slices),4);self.assertEqual(result.n_slices.tolist(),[2,2])


if __name__=='__main__':unittest.main()
