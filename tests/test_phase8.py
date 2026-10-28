"""Contrastes estrictos, aislamiento de errores y análisis sin alterar los runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import test_phase4 as fixtures
from src.batch_train import resolve_job, main as batch_main
from src.check_suite import verify_suite, verify_partitions
from src.checkpointing import load_resume
from src.compare_runs import supplementary_analysis, trainable_parameters
from src.configuration import load_config, project_path
from src.experiment import train
from src.model import BreastCancerCNN
from src.phase8 import audit_job, phase8_comparison, verify_phase8_suite


class Phase8Tests(unittest.TestCase):
    def setUp(self):
        # Los controles reales de clase se auditan en preflight, no son fixtures
        # implícitas de unit tests. Los tests de discrepancia aportan su snapshot.
        self.controls = patch('src.phase8.summary_paths', return_value=[])
        self.controls.start()
        self.addCleanup(self.controls.stop)

    def entries(self):
        return load_config('config/phase8/suite.json')['experiments']

    def test_three_strict_single_changes_and_verified_normalization(self):
        result = verify_suite('config/phase8/suite.json', synthetic=False)
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(len(result['models']), 3)
        for entry in self.entries():
            cfg = resolve_job(entry['config'], 'cpu')
            parent = resolve_job(entry['control_config'], 'cpu')
            changed = {k for k in cfg if cfg[k] != parent[k]} - {'RUN_NAME','COMPARISON_PARENT'}
            self.assertEqual(changed, set(entry['expected_changes']))
            self.assertEqual(cfg['EPOCHS'], 20)
            self.assertEqual(cfg['PATIENCE'], 8)
            self.assertTrue(cfg['EVALUATE_TRAIN'])
            self.assertEqual(cfg['STATS_PROVENANCE'], 'verified_train_fit')
            self.assertEqual(cfg['CHANNEL_MEANS'], parent['CHANNEL_MEANS'])
            self.assertLessEqual(max(cfg['CONV_CHANNELS']),128)

    def test_additional_hyperparameter_change_is_rejected(self):
        entry = self.entries()[0]
        cfg = resolve_job(entry['config'],'cpu')
        with self.assertRaisesRegex(ValueError,'DROPOUT'):
            audit_job(entry,dict(cfg,DROPOUT=.5))
        with self.assertRaisesRegex(ValueError,'Estadísticas'):
            audit_job(dict(entry,reference_statistics_sha256='wrong'),cfg)

    def test_one_bad_statistics_signature_does_not_hide_other_preflights(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=load_config('config/phase8/suite.json')
            suite['experiments'][0]['reference_statistics_sha256']='different'
            path=Path(tmp)/'suite.json';path.write_text(json.dumps(suite),encoding='utf-8')
            result=verify_phase8_suite(str(path),synthetic=False)
            self.assertFalse(result['passed'])
            self.assertEqual([r['id'] for r in result['errors']],['F8_SMALL_LR'])
            self.assertEqual(len(result['models']),2)

    def test_actual_control_snapshot_discrepancy_blocks_only_its_contrast(self):
        entry=self.entries()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'internals').mkdir()
            cfg=resolve_job(entry['control_config'],'cpu')
            cfg['CHANNEL_MEANS']=[.123,.234,.345]
            (root/'internals/config.json').write_text(json.dumps(cfg),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Control histórico incompatible'):
                audit_job(dict(entry,control_run=str(root)))
            self.assertEqual(audit_job(self.entries()[2])['id'],'F8_CONTROL_LR')

    def test_existing_wrong_phase_path_is_rejected_before_training(self):
        cfg=resolve_job(self.entries()[0]['config'],'cpu')
        tr=pd.read_csv(project_path(cfg['TRAIN_CSV']))
        va=pd.read_csv(project_path(cfg['VAL_CSV']))
        # El archivo existe, pero no representa EARLY: no basta con is_file().
        tr.loc[0,'path_early']=tr.loc[0,'path_pre']
        with patch('src.check_suite.pd.read_csv',side_effect=[tr,va]):
            with self.assertRaisesRegex(ValueError,'fases'):
                verify_partitions(cfg)

    def test_batch_rejects_affected_model_and_continues_without_training_gpu(self):
        from src.phase8 import audit_job as original
        def audit(entry,cfg):
            if entry['id']=='F8_SMALL_LR':
                raise ValueError('statistics discrepancy')
            return original(entry,cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sys.argv',['batch_train','--suite','config/phase8/suite.json','--device','cuda']), patch.object(
                    torch.version,'hip','mock_for_routing_only',create=True), patch(
                    'torch.cuda.is_available',return_value=True), patch('src.batch_train.check_device'), patch(
                    'src.batch_train.project_path',side_effect=lambda p:root if str(p)=='models' else project_path(p)), patch(
                    'src.batch_train.comparison',return_value=pd.DataFrame(columns=['Run'])), patch(
                    'src.batch_train.completed_run',return_value=None), patch('src.phase8.audit_job',side_effect=audit), patch(
                    'src.batch_train.train',side_effect=lambda cfg,resume_from=None:(root/cfg['RUN_NAME'],{'status':'mock_no_training'})) as training:
                batch_main()
            self.assertEqual(training.call_count,2)
            ids=[call.args[0]['RUN_NAME'] for call in training.call_args_list]
            self.assertEqual(ids,['F8_SMALL_DROPOUT','F8_CONTROL_LR'])
            folder=next(root.glob('BATCH_*'))
            self.assertIn('statistics discrepancy',(folder/'errors.jsonl').read_text(encoding='utf-8'))

    def test_parameter_count_matches_cnn_without_rng_consumption(self):
        for cfg in [dict(CONV_CHANNELS=[16,32,64],CONVS_PER_BLOCK=2),
                    dict(CONV_CHANNELS=[32,64,128],CONVS_PER_BLOCK=2),
                    dict(CONV_CHANNELS=[4,8],CONVS_PER_BLOCK=1,USE_BATCH_NORM=False,SPATIAL_GRID=2)]:
            model=BreastCancerCNN(cfg)
            expected=sum(p.numel() for p in model.parameters() if p.requires_grad)
            state=torch.get_rng_state().clone()
            self.assertEqual(trainable_parameters(model.config),expected)
            torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)

    def test_ap_uses_continuous_scores_and_history_is_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'internals').mkdir()
            pd.DataFrame(dict(label=[0,1,0,1],probability=[.1,.3,.2,.4])).to_csv(root/'internals/predictions_patient.csv',index=False)
            history=pd.DataFrame(dict(epoch=[1,2],train_eval_loss=[.6,.4],val_loss=[.7,.8]))
            result=supplementary_analysis(root,history,dict(recall=0,specificity=1))
            self.assertEqual(result['PR_AUC_AP_paciente'],1.)
            self.assertEqual(result['Balanced_accuracy_paciente'],.5)
            self.assertEqual(result['Epoca_min_val_loss'],1)
            self.assertIn('compatible',result['Evidencia_sobreajuste'])
            result=supplementary_analysis(root,history.drop(columns='train_eval_loss'),dict(recall=0,specificity=1))
            self.assertIsNone(result['Delta_train_eval_loss'])

    def test_missing_clinical_results_stay_pending_without_invented_auc(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=phase8_comparison(tmp)
            self.assertEqual(len(result),5)
            self.assertTrue(result.ROC_AUC_paciente.isna().all())
            self.assertEqual(sum(result.Estado.eq('pendiente_GPU')),3)

    def test_comparison_marks_a_different_control_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'RUN_control';(run/'internals').mkdir(parents=True)
            cfg=resolve_job('config/phase7/SPATIAL_SMALL.json','cpu')
            cfg['CHANNEL_MEANS']=[.123,.234,.345]
            path=run/'internals/config.json';path.write_text(json.dumps(cfg),encoding='utf-8')
            before=path.read_bytes()
            rows=pd.DataFrame([dict(Experimento='F7_SPATIAL_SMALL',Run=run.name,
                                    Grupo_comparable='train_completo',Estado='entrenamiento_completo')])
            with patch('src.compare_runs.comparison',return_value=rows):
                result=phase8_comparison(tmp)
            control=result[result.Experimento.eq('F7_SPATIAL_SMALL')].iloc[0]
            self.assertEqual(control.Comparabilidad,'config_difiere_revisar')
            self.assertIn('CHANNEL_MEANS',control.Diferencias_config)
            self.assertEqual(before,path.read_bytes())

    def test_small_lr_synthetic_checkpoints_graphs_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=fixtures.Phase4Tests().fixture(Path(tmp))
            cfg=resolve_job(self.entries()[0]['config'],'cpu')
            for key in ['TRAIN_CSV','VAL_CSV','DATA_DIR','RESULTS_DIR','IMAGE_SIZE','BATCH_SIZE','EPOCHS']:
                cfg[key]=fixture[key]
            cfg.update(DEVICE='cpu',SMOKE_TEST=True,SMOKE_PATIENTS_PER_CLASS=2,
                       EXPECTED_TRAIN_PATIENTS=4,EXPECTED_VAL_PATIENTS=4,
                       STATS_PATH=None,STATS_PROVENANCE='synthetic_fixture_only')
            directory,summary=train(cfg)
            self.assertTrue(summary['diagnostic_partial'])
            self.assertEqual(summary['epochs_completed'],2)
            for name in ['epoch_001.pt','epoch_002.pt','best_epoch.pt','loss_curve.png','loss_comparison.png',
                         'accuracy_curve.png','auc_curve.png','roc_curve.png','history.csv','config.py']:
                self.assertTrue((directory/name).is_file(),name)
            self.assertEqual(load_resume(directory/'epoch_001.pt',cfg)['epoch'],1)
