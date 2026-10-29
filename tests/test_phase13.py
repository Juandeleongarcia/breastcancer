"""F13: datos geométricos y clínica ficticia en temporales. Nunca entrena el dataset real."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src import phase13 as p
from src.configuration import canonical_csv_hash,load_config
from src.checkpointing import capture_rng
from src.model import BreastCancerCNN
from src.phase13_model import SpatialSmallMultimodal,architecture_report
from src.phase13_clinical import ClinicalTransform,feature_hash
from src.phase13_data import read_index,clinical_table,partition,MultimodalDataset,audit
from src.phase13_engine import train_job,evaluate,run_epoch,compatible_checkpoint
from src.phase13_reports import unique_directory,mean_history,recommend,history_plots,validate_patient_predictions
from src.phase13_inference import ImageTransform,load_bundle,predict_patient
from src.experiment import set_seed


class Phase13Tests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(2)
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup();torch.set_num_threads(self.threads)

    def fixture(self,size=16,epochs=2,smoke=True):
        folder=self.root/('data_'+str(size));folder.mkdir()
        grid=np.indices((size,size)).sum(axis=0)
        rows=[];patients=[]
        for fold in [-1,0,1,2,3,4]:
            split='test' if fold==-1 else 'train'
            for label in [0,1]:
                pid=f'TEST_ONLY_{fold+1}_{label}'
                target=folder/'dataset'/split/pid;target.mkdir(parents=True)
                patients.append(dict(pid=pid,pCR=label,split=split,test=int(fold==-1),
                    HR_HER2_STATUS=['HRposHER2neg','TripleNeg','HER2pos'][(fold+label)%3],
                    age=30+fold*5+label,tum_vol=9999.,post_treatment_RESPONSE=label))
                for z in [0,1]:
                    pre=(grid+17+fold*3+label*10+z)%160
                    arrays=[pre,pre+10+grid%9,pre+5+grid%5]
                    paths={}
                    for phase,array,key in zip(['PRE','EARLY','LATE'],arrays,['pre','early','late']):
                        image=target/f'{pid}_z{z:03d}_{phase}.png'
                        Image.fromarray(array.astype(np.uint8)).save(image)
                        paths['path_'+key]=image.relative_to(folder).as_posix()
                    rows.append(dict(patient_id=pid,sample_id=f'{pid}_z{z:03d}',fold=fold,split=split,pCR=label,**paths))
        frame=pd.DataFrame(rows);frame.to_csv(folder/'samples.csv',index=False)
        pd.DataFrame(patients).to_csv(folder/'patients.csv',index=False)
        protocol=dict(samples_csv=str(folder/'samples.csv'),patients_csv=str(folder/'patients.csv'),
            samples_sha256=canonical_csv_hash(folder/'samples.csv'),data_dir=str(folder))
        cfg=p.template();cfg.update(EPOCHS=epochs,DEVICE='cpu',TORCH_NUM_THREADS=2)
        if size!=256:cfg.update(IMAGE_SIZE=size,BATCH_SIZE=2,TECHNICAL_FIXTURE=True)
        campaign=p.prepare(self.root/'campaigns',smoke=smoke,protocol=protocol,config=cfg)
        return campaign,protocol,cfg

    def job(self,campaign):
        cfg,clinical=p.fit_job(campaign,0)
        root,state,base,dev,patients,external=p.checked(campaign)
        cfg['DEVICE']='cpu'
        contract=p.training_contract(state,cfg,clinical,cfg['TRAIN_CSV'],cfg['VAL_CSV'],patients)
        return cfg,clinical,patients,contract

    def test_exact_extractor_parameters_shapes_and_gradients(self):
        cfg=p.template();set_seed(42);original=BreastCancerCNN(cfg)
        set_seed(42);model=SpatialSmallMultimodal(cfg,7)
        self.assertTrue(all(torch.equal(v,model.features.state_dict()[k]) for k,v in original.features.state_dict().items()))
        report=architecture_report(cfg,7)
        self.assertEqual(report['original_parameters'],72593);self.assertEqual(report['parameters'],72600)
        self.assertEqual([c['out_channels'] for c in report['convolutions']],[16,16,32,32,64,64])
        self.assertEqual(report['batchnorm_layers'],6)
        image=torch.randn(1,3,256,256);clinical=torch.randn(1,7)
        logits=model(image,clinical);self.assertEqual(tuple(logits.shape),(1,1))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.classifier[1].weight.grad).all())
        self.assertGreater(float(model.classifier[1].weight.grad[:,64:].abs().sum()),0)
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.features.parameters() if v.grad is not None))
        with self.assertRaises(ValueError):model(image,torch.zeros(2,7))

    def test_clinical_imputation_categories_and_roundtrip_train_only(self):
        table=pd.DataFrame(dict(pid=['a','b','c','v'],age=[20.,40.,None,110.],
            HR_HER2_STATUS=['A','B',None,'VALIDATION_ONLY'],pCR=[0,1,0,1]))
        transform=ClinicalTransform.fit(table,['a','b','c'],['v'])
        self.assertEqual(transform.state['age_median'],30.)
        self.assertEqual(transform.state['age_mean'],30.)
        self.assertNotIn('VALIDATION_ONLY',transform.state['categories'])
        values=transform.transform(table,['c','v'])
        self.assertEqual(values[0,1],1);self.assertEqual(values[1,-1],1)
        self.assertEqual(values[0,-2],1)
        restored=ClinicalTransform(json.loads(json.dumps(transform.state)))
        np.testing.assert_array_equal(values,restored.transform(table,['c','v']))
        table.loc[3,'age']=115.
        self.assertEqual(ClinicalTransform.fit(table,['a','b','c'],['v']).state,transform.state)
        with self.assertRaises(ValueError):ClinicalTransform.fit(table,['a','v'],['v'])
        with self.assertRaises(ValueError):ClinicalTransform.fit(table,['a','a'])
        with self.assertRaises(ValueError):transform.transform(table,['absent'])

    def test_clinical_allowlist_and_external_labels_not_audited(self):
        campaign,protocol,_=self.fixture()
        dev,external=read_index(protocol)
        table=pd.read_csv(protocol['patients_csv'])
        table.loc[table.split.eq('test'),'pCR']=777
        table.to_csv(protocol['patients_csv'],index=False)
        result=audit(protocol)
        self.assertFalse(result['external_labels_analyzed']);self.assertEqual(result['reserved_patients'],2)
        clinical=clinical_table(protocol['patients_csv'],dev.patient_id.unique())
        self.assertEqual(set(clinical),{'pid','age','HR_HER2_STATUS'})
        self.assertFalse(set(clinical.pid)&external)
        changed=clinical.copy();changed['pCR']=1;changed['fold']=-1;changed['tum_vol']=123
        self.assertEqual(feature_hash(clinical,clinical.pid),feature_hash(changed,changed.pid))

    def test_official_folds_complete_and_no_patient_leakage(self):
        campaign,protocol,_=self.fixture()
        dev,external=read_index(protocol);seen=set()
        for fold in range(5):
            tr,va=partition(dev,fold)
            self.assertFalse(set(tr.patient_id)&set(va.patient_id))
            self.assertFalse(set(tr.patient_id)&external)
            self.assertEqual(len(tr),16);self.assertEqual(len(va),4)
            self.assertFalse(seen&set(va.patient_id));seen.update(va.patient_id)
        self.assertEqual(seen,set(dev.patient_id))
        with self.assertRaises(ValueError):partition(dev,-1)
        original=pd.read_csv(protocol['samples_csv']);original.loc[0,'fold']=0
        original.to_csv(protocol['samples_csv'],index=False)
        with self.assertRaises(ValueError):read_index(protocol)

    def test_train_only_image_stats_and_exact_dataset_preprocessing(self):
        campaign,protocol,_=self.fixture()
        opened=[];real=Image.open
        def observe(path,*args,**kwargs):opened.append(str(path));return real(path,*args,**kwargs)
        with patch('src.cv_data.Image.open',side_effect=observe):cfg,clinical,patients,contract=self.job(campaign)
        _,_,_,dev,_,external=p.checked(campaign);tr,va=partition(dev,0)
        self.assertTrue(opened)
        self.assertFalse(any('/test/' in path.replace('\\','/') for path in opened))
        self.assertFalse(any(pid in path for path in opened for pid in va.patient_id.unique()))
        dataset=MultimodalDataset(cfg['TRAIN_CSV'],cfg['DATA_DIR'],cfg,patients,ClinicalTransform(clinical))
        self.assertTrue(torch.equal(dataset[0]['clinical'],dataset[1]['clinical']))
        row=dataset.df.iloc[0]
        phases=np.stack([np.array(Image.open(Path(cfg['DATA_DIR'])/row[name])) for name in ['path_pre','path_early','path_late']])
        self.assertTrue(torch.equal(ImageTransform(cfg)(phases),dataset[0]['image']))
        state=load_config(campaign/'campaign.json');stats=load_config(Path(cfg['STATS_PATH']))
        self.assertEqual(stats['n_samples'],len(tr));self.assertEqual(stats['n_patients'],tr.patient_id.nunique())

    def test_evaluation_preserves_weights_bn_modes_and_all_rng(self):
        campaign,_,_=self.fixture();cfg,clinical,patients,contract=self.job(campaign)
        dataset=MultimodalDataset(cfg['TRAIN_CSV'],cfg['DATA_DIR'],cfg,patients,ClinicalTransform(clinical))
        model=SpatialSmallMultimodal(cfg,len(clinical['columns']));model.train()
        model.features[1].eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        modes=[m.training for m in model.modules()]
        generator=torch.Generator().manual_seed(7);rng=capture_rng(generator)
        result=evaluate(model,dataset,torch.nn.BCEWithLogitsLoss(),torch.device('cpu'),cfg,generator)
        self.assertEqual(result['patient']['n'],8)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertEqual(modes,[m.training for m in model.modules()])
        after=capture_rng(generator)
        self.assertEqual(rng['python'],after['python']);self.assertEqual(rng['numpy'],after['numpy'])
        self.assertTrue(torch.equal(rng['torch'],after['torch']));self.assertTrue(torch.equal(rng['loader'],after['loader']))

    def test_cpu_epoch_checkpoints_curves_and_exact_resume(self):
        campaign,_,_=self.fixture();cfg,clinical,patients,contract=self.job(campaign)
        full=unique_directory(self.root,'RUN_full');train_job(cfg,clinical,patients,contract,full)
        partial=unique_directory(self.root,'RUN_partial');summary=train_job(cfg,clinical,patients,contract,partial,stop_after_epoch=1)
        self.assertEqual(summary['status'],'interrupted')
        oldbytes=(partial/'epoch_001.pt').read_bytes()
        resumed=unique_directory(self.root,'RUN_resume')
        train_job(cfg,clinical,patients,contract,resumed,partial/'epoch_001.pt')
        a=torch.load(full/'epoch_002.pt',weights_only=True);b=torch.load(resumed/'epoch_002.pt',weights_only=True)
        self.assertTrue(all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items()))
        self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
        for key,values in a['optimizer_state_dict']['state'].items():
            for name,value in values.items():
                actual=b['optimizer_state_dict']['state'][key][name]
                self.assertTrue(torch.equal(value,actual) if isinstance(value,torch.Tensor) else value==actual)
        self.assertTrue(torch.equal(a['rng_state']['loader'],b['rng_state']['loader']))
        self.assertTrue(torch.equal(a['rng_state']['torch'],b['rng_state']['torch']))
        self.assertEqual(oldbytes,(partial/'epoch_001.pt').read_bytes())
        for key in ['train_eval_loss','val_loss','val_patient_auc','val_patient_recall','val_patient_f1']:
            self.assertEqual([r[key] for r in a['history']],[r[key] for r in b['history']])
        for name in ['config.py','history.csv','epoch_001.pt','epoch_002.pt','best_epoch.pt',
                     'loss_curve.png','accuracy_curve.png','auc_curve.png','roc_curve.png']:
            self.assertTrue((full/name).is_file(),name)
        best=torch.load(full/'best_epoch.pt',weights_only=True)
        history=pd.read_csv(full/'history.csv')
        self.assertEqual(best['epoch'],int(history.loc[history.val_patient_auc.idxmax(),'epoch']))
        with self.assertRaises(ValueError):compatible_checkpoint(partial/'epoch_001.pt',dict(contract,changed=True),cfg,torch.device('cpu'))
        with self.assertRaises(ValueError):compatible_checkpoint(partial/'epoch_001.pt',contract,cfg,torch.device('cuda'))

    def test_interrupted_epoch_replays_from_start_with_optimizer_rng(self):
        campaign,_,_=self.fixture(epochs=1);cfg,clinical,patients,contract=self.job(campaign)
        interrupted=unique_directory(self.root,'RUN_interrupt')
        real=run_epoch
        def stop(*args,**kwargs):
            real(*args,**kwargs)
            raise KeyboardInterrupt()
        with patch('src.phase13_engine.run_epoch',side_effect=stop):
            with self.assertRaises(KeyboardInterrupt):train_job(cfg,clinical,patients,contract,interrupted)
        checkpoint=p.recovery_checkpoint(interrupted)
        self.assertEqual(torch.load(checkpoint,weights_only=True)['epoch'],0)
        resumed=unique_directory(self.root,'RUN_replayed');train_job(cfg,clinical,patients,contract,resumed,checkpoint)
        fresh=unique_directory(self.root,'RUN_fresh');train_job(cfg,clinical,patients,contract,fresh)
        a=torch.load(resumed/'epoch_001.pt',weights_only=True);b=torch.load(fresh/'epoch_001.pt',weights_only=True)
        self.assertTrue(all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items()))
        self.assertTrue(a['optimizer_state_dict']['state'])

    def test_progressive_folds_skip_completed_and_smoke_cannot_recommend(self):
        campaign,_,_=self.fixture(epochs=1)
        p.run(campaign,[0,1],device='cpu')
        state=load_config(campaign/'campaign.json');old=state['folds']['0']['last_run']
        with patch('src.phase13.train_job',wraps=train_job) as trainer:
            result=p.run(campaign,[0,1,2],device='cpu')
            self.assertEqual(trainer.call_count,1)
        self.assertEqual(load_config(campaign/'campaign.json')['folds']['0']['last_run'],old)
        self.assertEqual(result['complete_folds'],0);self.assertTrue(result['excluded_diagnostics'])
        self.assertFalse((campaign/'recommended_final_training.json').exists())
        with self.assertRaises(ValueError):p.final_train(campaign,selected=True)

    def synthetic_completed_campaign(self):
        campaign,protocol,cfg=self.fixture(256,epochs=4,smoke=False)
        for fold in range(5):p.fit_job(campaign,fold)
        state=load_config(campaign/'campaign.json');dev,_=read_index(protocol)
        epochs=[1,2,3,4,2]
        for fold,best in enumerate(epochs):
            folder=campaign/f'fold_{fold}'
            run=unique_directory(folder/'runs','RUN_TEST_ONLY')
            details=run/'internals';details.mkdir()
            history=pd.DataFrame([dict(epoch=e,learning_rate=.0005 if e<3 else .00025,
                train_eval_loss=.8-.1*e,val_loss=.65+.03*e,val_patient_auc=1. if e==best else .5,
                val_slice_auc=1.,train_eval_patient_accuracy=.5,val_patient_accuracy=1.,
                train_eval_patient_auc=.5) for e in range(1,5)])
            history.to_csv(run/'history.csv',index=False)
            validation=partition(dev,fold)[1]
            prediction=validation.groupby('patient_id').agg(label=('pCR','first'),n_slices=('sample_id','size')).reset_index()
            prediction['probability']=.1+.8*prediction.label
            prediction.to_csv(details/'predictions_patient.csv',index=False)
            # Marcador sintético: el resumidor no necesita ni lee pesos del modelo.
            torch.save({'TEST_ONLY':True},run/'best_epoch.pt')
            p.write_json(details/'summary.json',dict(status='complete',SMOKE_TEST=False,diagnostic_partial=False,
                validation_complete=True,epochs_completed=4,best_epoch=best,stopped_early=False))
            state['folds'][str(fold)].update(status='complete',last_run=run.relative_to(campaign).as_posix())
        p.write_json(campaign/'campaign.json',state)
        return campaign,protocol,cfg

    def test_summary_oof_means_recommendation_and_incomplete_protection(self):
        campaign,protocol,cfg=self.synthetic_completed_campaign()
        result=p.summarize(campaign)
        self.assertEqual(result['complete_folds'],5);self.assertEqual(result['oof_patient_auc'],1.)
        self.assertEqual(result['oof_patient']['confusion_matrix'],[[5,0],[0,5]])
        rec=load_config(campaign/'recommended_final_training.json')
        self.assertEqual(rec['best_epochs'],[1,2,3,4,2]);self.assertEqual(rec['recommended_epochs'],2)
        self.assertEqual(rec['median'],2);self.assertFalse(rec['external_test_used'])
        self.assertEqual(len(rec['final_lr_schedule']),2)
        with self.assertRaises(ValueError):validate_patient_predictions(pd.DataFrame(dict(patient_id=['wrong'],label=[0],probability=[.5],n_slices=[2])),read_index(protocol)[0])
        means=mean_history({0:pd.DataFrame(dict(epoch=[1,2,3],train_eval_loss=[.5,.4,.3],val_loss=[.6,.7,.8])),
                             1:pd.DataFrame(dict(epoch=[1],train_eval_loss=[.7],val_loss=[.8]))})
        self.assertEqual(means.n_folds.tolist(),[2,1,1]);self.assertTrue(np.isnan(means.iloc[-1].train_std))
        self.assertEqual(means.train_mean.tolist(),[.6,.4,.3])

    def test_final_training_and_external_command_only_on_synthetic_data(self):
        campaign,protocol,cfg=self.synthetic_completed_campaign()
        with self.assertRaises(ValueError):p.final_train(campaign,selected=False)
        with self.assertRaises(ValueError):p.external_test('DO_NOT_READ',confirm=False)
        final=p.final_train(campaign,device='cpu',selected=True)
        manifest=load_config(final/'final_training.json')
        self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['external_test_status'],'pending')
        bundle=load_bundle(final/manifest['checkpoint'])
        self.assertEqual(bundle['config']['EPOCHS'],2);self.assertEqual(bundle['config']['TRAINING_SCOPE'],'final_development')
        row=read_index(protocol)[0].iloc[0]
        phases=np.stack([np.array(Image.open(Path(protocol['data_dir'])/row[k])) for k in ['path_pre','path_early','path_late']])
        pred=predict_patient(bundle,np.stack([phases,phases]),dict(age=None,HR_HER2_STATUS='UNSEEN'))
        self.assertGreaterEqual(pred['patient_probability'],0);self.assertLessEqual(pred['patient_probability'],1)
        self.assertEqual(pred['patient_probability'],float(pred['slices'].groupby('patient_id').probability.mean().iloc[0]))
        with self.assertRaises(ValueError):predict_patient(bundle,np.stack([phases]),dict(age=30,HR_HER2_STATUS='A',pCR=1))
        # Se evalúa únicamente la cohorte ficticia de este TemporaryDirectory.
        report=p.external_test(final,confirm=True,device='cpu')
        self.assertTrue((report/'roc_curve.png').exists())
        self.assertEqual(load_config(report/'metrics.json')['patient']['n'],2)
        with self.assertRaises(ValueError):p.external_test(final,confirm=True)

    def test_corrupt_profiles_or_clinical_changes_block_resume(self):
        campaign,protocol,_=self.fixture();self.job(campaign)
        patients=pd.read_csv(protocol['patients_csv'])
        patients.loc[patients.split.eq('train'),'age']+=1
        patients.to_csv(protocol['patients_csv'],index=False)
        with self.assertRaises(ValueError):p.checked(campaign)

    def test_baseline_comparison_reads_results_without_weights_or_writes(self):
        import hashlib
        campaign,protocol,cfg=self.synthetic_completed_campaign()
        baseline=self.root/'baseline';baseline.mkdir()
        basecfg=load_config('config/phase9/SPATIAL_SMALL.json');basecfg.update(EPOCHS=4,TORCH_NUM_THREADS=2)
        state=dict(model='SPATIAL_SMALL',diagnostic=False,protocol=protocol,folds={})
        current=load_config(campaign/'campaign.json')
        for fold in range(5):
            source=campaign/current['folds'][str(fold)]['last_run']
            folder=baseline/f'fold_{fold}';run=folder/'runs'/f'RUN_BASE_TEST_ONLY_{fold}'
            (run/'internals').mkdir(parents=True)
            for name in ['train.csv','validation.csv']:shutil.copy2(campaign/f'fold_{fold}'/name,folder/name)
            shutil.copy2(source/'history.csv',run/'history.csv')
            for name in ['summary.json','predictions_patient.csv']:
                shutil.copy2(source/'internals'/name,run/'internals'/name)
            state['folds'][str(fold)]=dict(status='complete',last_run=f'/class/absent/fold_{fold}/runs/{run.name}')
        p.write_json(baseline/'campaign.json',state);p.write_json(baseline/'template.json',basecfg)
        before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in baseline.rglob('*') if f.is_file()}
        with patch('torch.load',side_effect=AssertionError('No abrir pesos para comparar')):
            result=p.compare(campaign,baseline)
        self.assertEqual(result['auc_mean_difference'],0.)
        self.assertFalse(result['baseline_checkpoint_required'])
        after={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in baseline.rglob('*') if f.is_file()}
        self.assertEqual(before,after)

    def test_plot_limits_do_not_clip_original_history_or_hide_overfitting(self):
        import matplotlib.figure
        frame=pd.DataFrame(dict(epoch=[1,2],train_eval_loss=[.4,.3],val_loss=[1.2,1.4],
            train_eval_patient_auc=[.6,.7],val_patient_auc=[.55,.52],
            train_eval_patient_accuracy=[.7,.8],val_patient_accuracy=[.7,.7]))
        saved={};real=matplotlib.figure.Figure.savefig
        def observe(fig,path,*args,**kwargs):
            saved[Path(path).name]=dict(ylim=fig.axes[0].get_ylim(),values=[line.get_ydata().copy() for line in fig.axes[0].lines])
            return real(fig,path,*args,**kwargs)
        with patch.object(matplotlib.figure.Figure,'savefig',new=observe):history_plots(frame,self.root/'plots',(0,1))
        self.assertEqual(saved['loss_curve_limited.png']['ylim'],(0.,1.))
        self.assertGreater(saved['loss_curve.png']['ylim'][1],1.4)
        np.testing.assert_array_equal(saved['loss_curve.png']['values'][1],[1.2,1.4])
        self.assertEqual(frame.val_loss.tolist(),[1.2,1.4])

    def test_cli_flags_and_all_subcommand_help(self):
        for command in ['check','audit','prepare','smoke','run','status','summarize','recommend','plots','compare','final-train','external-test']:
            with patch('sys.argv',['phase13',command,'--help']),redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit) as error:p.main()
                self.assertEqual(error.exception.code,0)
                self.assertIn(command,output.getvalue())


    def test_campaign_lock_blocks_overlap_and_releases_after_error(self):
        campaign,_,_=self.fixture()
        with p.campaign_lock(campaign):
            with self.assertRaises(RuntimeError):
                with p.campaign_lock(campaign):pass
        with self.assertRaises(KeyboardInterrupt):
            with p.campaign_lock(campaign):raise KeyboardInterrupt()
        with p.campaign_lock(campaign):pass


if __name__=='__main__':unittest.main()
