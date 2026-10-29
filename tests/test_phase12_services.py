from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from phase12_fixtures import study_files,png_bytes,checkpoint_fixture,campaign_fixture,fitted_config
from app_services.studies import study_from_uploads,study_from_files,synthetic_study,display_image
from app_services.preprocessing import TrainingTransform
from app_services.architectures import architecture_profile
from app_services.inference import load_checkpoint,predict_study
from app_services.results import read_history,aggregate_histories,read_campaign
from app_services.settings import load_settings,load_inference_plan
from app_components.charts import slice_stack,mean_loss_curve,confusion_figure


class Phase12ServiceTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(2)

    def tearDown(self):torch.set_num_threads(self.threads)

    def test_triplets_numeric_sort_and_zip_never_extract(self):
        files=study_files()
        study=study_from_files(list(reversed(files)))
        self.assertEqual(study.slice_indices,(0,1,2))
        self.assertEqual(study.phases.shape,(3,3,256,256))
        self.assertFalse(study.phases.flags.writeable)
        stream=BytesIO()
        with ZipFile(stream,'w') as archive:
            for name,data in files:archive.writestr('study/'+name,data)
        zipped=study_from_uploads([('study.zip',stream.getvalue())])
        self.assertEqual(zipped.fingerprint,study.fingerprint)

    def test_reject_missing_duplicate_patient_dimensions_rgb_and_traversal(self):
        good=study_files(n=1)
        bad=[good[:-1],good+[good[0]],good+study_files(n=1,patient='OTHER'),
             [(good[0][0],png_bytes(np.zeros((128,128),dtype=np.uint8))),*good[1:]],
             [(good[0][0],png_bytes(np.zeros((256,256,3),dtype=np.uint8))),*good[1:]],
             [('../'+good[0][0],good[0][1]),*good[1:]]]
        for files in bad:
            with self.subTest(files=[name for name,_ in files]):
                with self.assertRaises(ValueError):study_from_files(files)
        with self.assertRaises(ValueError):study_from_uploads([('broken.zip',b'not zip')])
        stream=BytesIO()
        with ZipFile(stream,'w') as archive:archive.writestr('../escape.png',good[0][1])
        with self.assertRaises(ValueError):study_from_uploads([('traversal.zip',stream.getvalue())])

    def test_preprocessing_bit_equal_to_original_dataset_and_display_is_separate(self):
        from src.dataset import BreastDCEDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=study_files(n=1)
            for name,data in files:(root/name).write_bytes(data)
            pd.DataFrame([dict(patient_id='TEST_ONLY',sample_id='TEST_ONLY_z000',pCR=0,
                path_pre=files[0][0],path_early=files[1][0],path_late=files[2][0])]).to_csv(root/'samples.csv',index=False)
            study=study_from_files(files);cfg=fitted_config()
            expected=BreastDCEDataset(root/'samples.csv',root,config=cfg)[0]['image']
            actual=TrainingTransform(cfg)(study.phases[0])
            self.assertTrue(torch.equal(actual,expected))
            fingerprint=study.fingerprint
            view=display_image(study.phases[0,0],.3,1.8)
            self.assertTrue((view>=0).all() and (view<=1).all())
            self.assertEqual(study.fingerprint,fingerprint)
            self.assertTrue(torch.equal(actual,TrainingTransform(cfg)(study.phases[0])))

    @unittest.skipUnless(os.environ.get('BREASTDCEDL_TEST_CHECKPOINTS')=='1','Arquitecturas aplazadas; base visual independiente del modelo')
    def test_actual_architectures_and_parameter_counts(self):
        small=architecture_profile('SPATIAL_SMALL');four=architecture_profile('F11_CNN4')
        self.assertEqual(small['parameters'],72593);self.assertEqual(four['parameters'],73913)
        self.assertEqual(sum(s['parameters'] for s in small['stages']),72593)
        self.assertEqual(sum(s['parameters'] for s in four['stages']),73913)
        self.assertEqual(four['stages'][4]['shape'],'64 × 32 × 32')

    @unittest.skipUnless(os.environ.get('BREASTDCEDL_TEST_CHECKPOINTS')=='1','Integración de pesos aplazada; habilitar solo al revisar esa fase')
    def test_both_checkpoint_types_exact_prediction_aggregation_no_training(self):
        study=study_from_files(study_files(n=3))
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ['SPATIAL_SMALL','F11_CNN4']:
                with self.subTest(kind=kind):
                    filename,cfg=checkpoint_fixture(tmp,kind)
                    before=filename.read_bytes()
                    bundle=load_checkpoint(filename,kind)
                    state={k:v.clone() for k,v in bundle.model.state_dict().items()}
                    result=predict_study(study,bundle)
                    self.assertEqual(result['patient_probability'],.5)
                    self.assertEqual(result['no_pcr_probability'],.5)
                    self.assertEqual(len(result['slices']),3)
                    self.assertEqual(result['threshold'],cfg['CLASSIFICATION_THRESHOLD'])
                    self.assertEqual(result['patient_probability'],result['slices'].groupby('patient_id').probability.mean().iloc[0])
                    self.assertTrue(all(torch.equal(state[k],v) for k,v in bundle.model.state_dict().items()))
                    self.assertEqual(filename.read_bytes(),before)
                    with self.assertRaisesRegex(ValueError,'sintética'):predict_study(synthetic_study(),bundle)
                    with self.assertRaises(ValueError):load_checkpoint(filename,'F11_CNN4' if kind=='SPATIAL_SMALL' else 'SPATIAL_SMALL')

    @unittest.skipUnless(os.environ.get('BREASTDCEDL_TEST_CHECKPOINTS')=='1','Integración de pesos aplazada; habilitar solo al revisar esa fase')
    def test_missing_stats_wrong_weights_incomplete_and_unknown_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename,_=checkpoint_fixture(tmp)
            original=torch.load(filename,weights_only=True)
            for edit in ['stats','weights','partial','architecture','missing']:
                payload=torch.load(filename,weights_only=True)
                if edit=='stats':payload['config']['STATS_PROVENANCE']='requires_fit'
                if edit=='weights':payload['model_state_dict'].pop(next(iter(payload['model_state_dict'])))
                if edit=='partial':payload['epoch_incomplete']=True
                if edit=='architecture':payload['config']['ARCHITECTURE']='unknown'
                if edit=='missing':payload['config'].pop('DROPOUT')
                bad=Path(tmp)/('bad_'+edit+'.pt');torch.save(payload,bad)
                with self.subTest(edit=edit):
                    with self.assertRaises(ValueError):load_checkpoint(bad)

    @unittest.skipUnless(os.environ.get('BREASTDCEDL_TEST_CHECKPOINTS')=='1','Integración de pesos aplazada; habilitar solo al revisar esa fase')
    def test_self_contained_normalization_and_companion_must_not_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename,cfg=checkpoint_fixture(tmp)
            cfg['DROPOUT']=.99
            companion=Path(tmp)/'config.json';companion.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'contradice'):load_checkpoint(filename,config_file=companion)
            bundle=load_checkpoint(filename)
            self.assertEqual(bundle.config['CHANNEL_STDS'],[.15,.12,.12])

    def test_epoch_means_intersection_counts_and_sample_std_no_filling(self):
        lengths=[3,2,1,3,2]
        histories={fold:pd.DataFrame(dict(epoch=range(1,n+1),train_eval_loss=np.arange(n)+fold,
                       val_loss=np.arange(n)+2*fold)) for fold,n in enumerate(lengths)}
        all_values=aggregate_histories(histories)
        self.assertEqual(all_values.n_folds.tolist(),[5,4,2])
        self.assertEqual(all_values.train_mean.tolist(),[2.,3.,3.5])
        self.assertAlmostEqual(all_values.iloc[0].train_std,np.std([0,1,2,3,4],ddof=1))
        common=aggregate_histories(histories,'common')
        self.assertEqual(common.epoch.tolist(),[1])
        self.assertTrue(aggregate_histories({0:histories[0]},'common').empty)
        one=aggregate_histories({0:histories[0]})
        self.assertTrue(one.train_std.isna().all())
        figure=mean_loss_curve(all_values)
        self.assertTrue(len(figure.data)>2)

    def test_history_duplicates_and_infinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'history.csv'
            pd.DataFrame(dict(epoch=[1,1],val_loss=[.2,.3])).to_csv(filename,index=False)
            with self.assertRaises(ValueError):read_history(filename)
            pd.DataFrame(dict(epoch=[1],val_loss=[float('inf')])).to_csv(filename,index=False)
            with self.assertRaises(ValueError):read_history(filename)

    def test_five_fold_reader_oof_confusion_portable_paths_and_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=campaign_fixture(tmp)
            state=json.loads((campaign/'campaign.json').read_text())
            for fold in range(5):state['folds'][str(fold)]['last_run']=f'/class/old/location/fold_{fold}/runs/RUN_fixture_{fold}'
            (campaign/'campaign.json').write_text(json.dumps(state))
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tmp).rglob('*') if p.is_file()}
            report=read_campaign(campaign,'SPATIAL_SMALL')
            self.assertTrue(report.complete,[f.issue for f in report.folds])
            self.assertEqual(report.metrics['patients'],10)
            self.assertEqual(report.metrics['auc_mean'],1.)
            self.assertEqual(report.metrics['auc_std'],0.)
            self.assertEqual(report.oof_metrics['confusion_matrix'],[[5,0],[0,5]])
            self.assertEqual(report.metrics['oof_auc'],1.)
            figure=confusion_figure(report.oof_metrics)
            self.assertEqual(np.asarray(figure.data[0].z).shape,(2,2))
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tmp).rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_incomplete_and_corrupt_predictions_never_become_complete_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=campaign_fixture(tmp,completed=2)
            report=read_campaign(campaign)
            self.assertEqual(len(report.completed),2,[f.issue for f in report.folds])
            self.assertFalse(report.complete);self.assertTrue(report.oof.empty)
            self.assertIsNone(report.metrics.get('oof_auc'))
            filename=campaign/'fold_1/runs/RUN_fixture_1/internals/predictions_patient.csv'
            frame=pd.read_csv(filename);pd.concat([frame,frame.iloc[:1]]).to_csv(filename,index=False)
            report=read_campaign(campaign)
            self.assertEqual(len(report.completed),1)
            self.assertTrue(report.oof.empty)

    def test_source_leakage_or_diagnostic_results_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=campaign_fixture(tmp)
            filename=campaign/'fold_0/validation.csv'
            frame=pd.read_csv(filename);frame['fold']=-1;frame.to_csv(filename,index=False)
            report=read_campaign(campaign)
            self.assertFalse(report.complete)
            self.assertEqual(len(report.completed),4)
            self.assertIn('oficiales',report.folds[0].issue)
        with tempfile.TemporaryDirectory() as tmp:
            campaign=campaign_fixture(tmp)
            state=json.loads((campaign/'campaign.json').read_text());state['diagnostic']=True
            (campaign/'campaign.json').write_text(json.dumps(state))
            report=read_campaign(campaign)
            self.assertEqual(len(report.completed),0)

    def test_schematic_volume_synchronization_and_no_anatomy_metadata(self):
        study=synthetic_study();before=study.fingerprint
        figure,count=slice_stack(study,1,7)
        self.assertLessEqual(count,13)
        self.assertTrue(np.equal(figure.data[-1].z,7).all())
        self.assertIn('sin mm',figure.layout.scene.zaxis.title.text)
        self.assertEqual(study.fingerprint,before)

    def test_settings_empty_and_unknown_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'app.json'
            filename.write_text(json.dumps({'models':{'SPATIAL_SMALL':{'checkpoint':'','campaign':''}}}))
            settings=load_settings(filename)
            self.assertEqual(settings['F11_CNN4'].checkpoint,'')
            filename.write_text(json.dumps({'models':{'PRETRAINED':{}}}))
            with self.assertRaises(ValueError):load_settings(filename)

    def test_malformed_json_objects_are_rejected_with_clear_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'settings.json';filename.write_text('[]')
            with self.assertRaises(ValueError):load_settings(filename)
            campaign=campaign_fixture(tmp)
            (campaign/'campaign.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'objeto JSON'):read_campaign(campaign)

    def test_inference_plan_is_disabled_independently_of_candidate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'settings.json'
            filename.write_text(json.dumps({'models':{'F11_CNN4':{'checkpoint':'DO_NOT_READ.pt'}}}))
            plan=load_inference_plan(filename)
            self.assertFalse(plan.enabled);self.assertEqual(plan.architecture,'')
            filename.write_text(json.dumps({'models':{},'inference':{'enabled':True}}))
            with self.assertRaises(ValueError):load_inference_plan(filename)

    def test_results_of_unregistered_architecture_do_not_require_model_factory(self):
        from src.cv_data import artifact_hash
        with tempfile.TemporaryDirectory() as tmp:
            campaign=campaign_fixture(tmp)
            template=json.loads((campaign/'template.json').read_text())
            template.update(ARCHITECTURE='FUTURE_CNN_TEST_ONLY',CONV_CHANNELS=[32,64,128])
            (campaign/'template.json').write_text(json.dumps(template))
            state=json.loads((campaign/'campaign.json').read_text())
            state['model']='EXPERIMENT_WITH_ANY_NAME'
            state['template_sha256']=artifact_hash(campaign/'template.json')
            (campaign/'campaign.json').write_text(json.dumps(state))
            for fold in range(5):
                filename=campaign/f'fold_{fold}/runs/RUN_fixture_{fold}/internals/config.json'
                cfg=json.loads(filename.read_text())
                cfg.update(ARCHITECTURE='FUTURE_CNN_TEST_ONLY',CONV_CHANNELS=[32,64,128])
                filename.write_text(json.dumps(cfg))
            report=read_campaign(campaign)
            self.assertTrue(report.complete,[f.issue for f in report.folds])
            self.assertEqual(report.architecture,'FUTURE_CNN_TEST_ONLY')
            self.assertEqual(report.metrics['patients'],10)


    def test_visual_workspace_reads_one_campaign_and_ignores_legacy_models(self):
        from app_services.workspace import load_workspace
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'visual.json'
            filename.write_text(json.dumps({'results':{'campaign':'LOCAL_RESULTS'},
                'models':{'unknown':{'checkpoint':'DO_NOT_READ.pt'}},
                'inference':{'enabled':True},'integration':{'enabled':True}}))
            workspace=load_workspace(filename)
            self.assertEqual(workspace.results_campaign,'LOCAL_RESULTS')
            self.assertEqual(set(vars(workspace)),{'results_campaign'})
            for data in [[],{'results':[]},{'results':{'campaign':42}}]:
                filename.write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_workspace(filename)

    def test_conceptual_figure_has_functions_without_architecture_details(self):
        from app_components.charts import CNN_CONCEPTS,conceptual_cnn_figure
        figure=conceptual_cnn_figure(2)
        self.assertEqual(tuple(figure.data[0].text),tuple(label for label,_ in CNN_CONCEPTS))
        self.assertEqual(figure.data[0].marker.color[2],'#42D3C4')
        payload=figure.to_json()
        for forbidden in ['SPATIAL_SMALL','F11_CNN4','72593','73913','32 × 32','parámetros']:
            self.assertNotIn(forbidden,payload)


if __name__=='__main__':unittest.main()
