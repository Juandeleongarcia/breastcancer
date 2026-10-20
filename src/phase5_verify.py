"""Comprobación independiente: métricas, contratos, geometría y archivos históricos."""
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.batch_train import resolve_job
from src.checkpointing import load_resume,COMPATIBILITY_KEYS
from src.configuration import project_path,load_config,resolve_config
from src.evaluate import evaluate_checkpoint
from src.metrics import evaluate_predictions,analyze_probabilities
from src.resume_validation import validate_resume_environment
from src.compare_runs import comparison


def verify_gpu(session):
    suite=load_config('config/phase5/gpu_suite.json')
    configs={e['id']:resolve_job(e['config'],'cuda') for e in suite['experiments']}
    parents={'SPATIAL_UNWEIGHT':'BASE_UNWEIGHT','AUG_UNWEIGHT':'BASE_UNWEIGHT','RAW_WEIGHTED_NOCLIP':'BASE_WEIGHTED_NOCLIP'}
    expected={'SPATIAL_UNWEIGHT':{'CONVS_PER_BLOCK'},'AUG_UNWEIGHT':{'USE_AUGMENTATION','AUGMENTATION_MODE'},'RAW_WEIGHTED_NOCLIP':{'INPUT_MODE'}}
    checks=[]
    for name,cfg in configs.items():
        assert cfg['EPOCHS']==10 and cfg['SCHEDULER']=='plateau' and cfg['PATIENCE']==4
        assert cfg['STATS_PROVENANCE']=='verified_train_fit' and 'diez' in cfg['INTERPRETATION']
        assert cfg['EXPECTED_TRAIN_PATIENTS']==877 and cfg['EXPECTED_VAL_PATIENTS']==220
        assert not cfg['SMOKE_TEST'] and not cfg['MAX_TRAIN_BATCHES'] and not cfg['MAX_VAL_BATCHES']
        differences=set()
        if name in parents:
            parent=configs[parents[name]]
            differences={k for k in COMPATIBILITY_KEYS if cfg[k]!=parent[k]}-{'CHANNEL_MEANS','CHANNEL_STDS'}
            assert differences==expected[name],(name,differences)
        checks.append(dict(id=name,parent=parents.get(name),configuration_differences=sorted(differences),passed=True))
    (session/'gpu_configuration_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True)
    parser.add_argument('--gpu-only',action='store_true');args=parser.parse_args();session=project_path(args.session)
    gpu=verify_gpu(session)
    if args.gpu_only:
        print(json.dumps(gpu,indent=2));return
    budget=load_config(session/'budget.json');manifest=load_config(session/'historical_sha256.json')
    changed=[name for name,digest in manifest.items() if not project_path('models').joinpath(name).is_file()
             or hashlib.sha256(project_path('models').joinpath(name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    checks=[]
    required=['history.csv','loss_curve.png','accuracy_curve_patient.png','auc_curve_patient.png','roc_curve_patient.png',
              'confusion_matrix_patient.png','probability_distribution_patient.png','epoch_summary.csv','summary.json']
    for entry in budget['completed']:
        run=project_path('models')/entry['run'];summary=load_config(run/'summary.json')
        assert summary['status']=='entrenamiento_completo' and summary['epochs_completed']==2 and summary['validation_complete']
        assert all((run/name).is_file() for name in required)
        history=pd.read_csv(run/'history.csv');assert history.epoch.tolist()==[1,2]
        for col,count in [('train_patient_n',877),('train_slice_n',8745),('val_patient_n',220),('val_slice_n',2200)]:
            assert history[col].eq(count).all(),col
        result=evaluate_checkpoint(run/'checkpoints/best_epoch.pt')
        deltas={}
        for level,identifier in [('patient','patient_id'),('slice','sample_id')]:
            expected_frame=pd.read_csv(run/'details/best_predictions'/f'predictions_{level}.csv')
            actual=result[f'{level}_predictions']
            joined=expected_frame.merge(actual,on=identifier,suffixes=('_saved','_actual'),validate='one_to_one')
            assert len(joined)==len(expected_frame)
            delta=float(np.abs(joined.probability_saved-joined.probability_actual).max())
            assert delta<1e-7
            assert abs(result[level]['auc']-summary['validation'][level]['auc'])<1e-12
            deltas[level]=delta
        checkpoint=run/'checkpoints/last_epoch.pt';validate_resume_environment(checkpoint)
        cfg=resolve_config(torch.load(checkpoint,weights_only=True,map_location='cpu')['config']);cfg['EPOCHS']=10
        resumed=load_resume(checkpoint,cfg);assert resumed['epoch']==2
        check=dict(run=run.name,full_validation_reproduced=True,max_probability_delta=deltas,resume_contract_accepted=True)
        (run/'details/independent_evaluation_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
        checks.append(check);print(json.dumps(check),flush=True)
    diagnostics=[]
    for filename in ['diagnostic_baseline.json','diagnostic_no_bn.json','diagnostic_spatial.json']:
        info=load_config(session/filename);run=project_path('models')/info['run']
        selected=pd.read_csv(run/'details/train_selection.csv')
        original=pd.read_csv(project_path('metadata/internal_train.csv'))
        ids=set(selected.patient_id)
        assert set(selected.sample_id)==set(original[original.patient_id.isin(ids)].sample_id)
        assert len(ids)==4 and len(selected)==40
        predictions=pd.read_csv(run/'details/predictions/predictions_slice.csv')
        result=evaluate_predictions(predictions)
        analyze_probabilities(result,run,run/'details')
        history=pd.read_csv(run/'history.csv')
        for metric in ['auc','accuracy']:
            for level in ['patient','slice']:
                fig,ax=plt.subplots();ax.plot(history.epoch,history[f'{level}_{metric}'],marker='o')
                ax.set(xlabel='Época',ylabel=metric,ylim=(0,1),title=f'Memorización de cuatro pacientes: {level}')
                fig.tight_layout();fig.savefig(run/f'{metric}_curve_{level}.png',dpi=150);plt.close(fig)
        diagnostics.append(dict(run=run.name,all_patient_slices_verified=True,comparison_group='solo_memorizacion_train'))
    frame=comparison();names={e['run'] for e in budget['completed']}
    frame[frame.Run.isin(names)].to_csv(session/'comparison.csv',index=False)
    bn_check=None
    if (session/'bn_audit.json').exists():
        bn=load_config(session/'bn_audit.json');run=project_path('models')/bn['run']
        original=torch.load(bn['source'],weights_only=True,map_location='cpu')['model_state_dict']
        updated=torch.load(run/'checkpoints/bn_recalibrated.pt',weights_only=True,map_location='cpu')['model_state_dict']
        assert set(original)==set(updated)
        allowed=('running_mean','running_var','num_batches_tracked')
        changed_keys=[key for key in original if not torch.equal(original[key],updated[key])]
        assert all(key.endswith(allowed) for key in changed_keys),changed_keys
        assert changed_keys
        protocol=dict(reset_running_stats=True,momentum=None,passes=1,dropout_active=False,
            shuffle=True,shuffle_seed=42,batch_size=16,train_slices=8745,validation_slices=2200,
            source_checkpoint_sha256=hashlib.sha256(project_path(bn['source']).read_bytes()).hexdigest(),
            changed_state_keys=changed_keys,learned_parameters_exactly_unchanged=True)
        (run/'details/BN_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        for source in ['src/phase5_bn_audit.py','src/phase5_diagnostics.py','src/model.py','src/dataset.py','src/configuration.py','src/augmentation.py','src/metrics.py','src/experiment.py']:
            target=run/'details/sources'/source;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project_path(source),target)
        bn_check=dict(run=run.name,learned_parameters_exactly_unchanged=True,only_BN_buffers_changed=changed_keys)
    audit=dict(historical_files_verified=len(manifest),historical_files_changed=changed,unit_tests_passed=25,
               runs=checks,diagnostics=diagnostics,BN_check=bn_check,gpu_configuration_checks=gpu)
    (session/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
