"""Campaña CPU con diagnósticos separados y deadline conservador de 75 minutos."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
from torch.utils.data import DataLoader

from src.batch_train import resolve_job
from src.configuration import load_config, project_path
from src.checkpointing import contract
from src.experiment import (set_seed,select_device,make_datasets,create_optimizer,
                            create_criterion,run_epoch,train)
from src.model import BreastCancerCNN
from src.compare_runs import comparison


def measure(cfg, deadline):
    set_seed(cfg['SEED']);device=select_device(cfg)
    model=BreastCancerCNN(cfg);optimizer=create_optimizer(model,cfg);criterion=create_criterion(cfg,device)
    train_ds,val_ds=make_datasets(cfg);times={}
    for name,ds,opt in [('train',train_ds,optimizer),('validation',val_ds,None)]:
        loader=DataLoader(ds,batch_size=cfg['BATCH_SIZE'],shuffle=False,num_workers=0)
        started=time.time();result=run_epoch(model,loader,criterion,device,optimizer=opt,max_batches=8,deadline=deadline)
        elapsed=time.time()-started
        times[name]=dict(seconds=elapsed,samples=result['slice']['n'],total_samples=len(ds))
    times['estimated_two_epoch_seconds']=2*sum(r['seconds']*r['total_samples']/r['samples'] for r in times.values())+20
    times['note']='Benchmark técnico de tiempo; no resultado de generalización. Incluye augmentations si están activas.'
    return times


def prepare(session):
    budget=load_config(session/'budget.json');entries=[]
    for entry in load_config('config/phase5/today_suite.json')['experiments']:
        cfg=resolve_job(entry['config'],'cpu');entries.append(dict(entry,config_resolved=cfg,contract=contract(cfg)))
    with (session/'plan.json').open('x',encoding='utf-8') as output:json.dump(entries,output,indent=2)
    train_frame=pd.read_csv(project_path(entries[0]['config_resolved']['TRAIN_CSV']))
    val_frame=pd.read_csv(project_path(entries[0]['config_resolved']['VAL_CSV']))
    keys=['path_pre','path_early','path_late']
    joined=pd.concat([train_frame,val_frame],ignore_index=True)
    audit=dict(train_patients=int(train_frame.patient_id.nunique()),val_patients=int(val_frame.patient_id.nunique()),
        train_slices=len(train_frame),val_slices=len(val_frame),
        shared_patient_ids=len(set(train_frame.patient_id)&set(val_frame.patient_id)),
        conflicting_patient_labels=int(joined.groupby('patient_id').pCR.nunique().gt(1).sum()),
        conflicting_triplet_path_labels=int(joined.groupby(keys).pCR.nunique().gt(1).sum()),
        shared_triplet_paths=len(set(map(tuple,train_frame[keys].values))&set(map(tuple,val_frame[keys].values))),
        train_class_counts=train_frame.drop_duplicates('patient_id').pCR.value_counts().to_dict(),
        validation_class_counts=val_frame.drop_duplicates('patient_id').pCR.value_counts().to_dict(),
        note='Consistencia interna de etiquetas y rutas; no verifica veracidad clínica ni identidad de píxeles en rutas distintas.')
    assert audit['shared_patient_ids']==audit['conflicting_patient_labels']==audit['conflicting_triplet_path_labels']==audit['shared_triplet_paths']==0
    (session/'label_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    for entry in entries[:2]:
        result=measure(entry['config_resolved'],budget['deadline_unix']-30)
        (session/f"benchmark_{entry['id']}.json").write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(dict(experiment=entry['id'],**result)),flush=True)


def execute(session,identifier,decision):
    budget=load_config(session/'budget.json')
    if any(e['experiment']==identifier for e in budget['completed']):raise ValueError('Ya ejecutado; no repetir ni sobrescribir')
    entry=next(e for e in load_config(session/'plan.json') if e['id']==identifier)
    cfg=entry['config_resolved']
    assert contract(cfg)==entry['contract'],'El código cambió después de congelar el plan'
    assert cfg['EPOCHS']==2 and not cfg['SMOKE_TEST'] and not cfg['MAX_TRAIN_BATCHES'] and not cfg['MAX_VAL_BATCHES'] and not cfg.get('TRAIN_PATIENTS_PER_CLASS')
    benchmark=session/f'benchmark_{identifier}.json'
    if not benchmark.exists():
        if budget['deadline_unix']-time.time()<180:return
        benchmark.write_text(json.dumps(measure(cfg,budget['deadline_unix']-30),indent=2),encoding='utf-8')
    estimate=load_config(benchmark)['estimated_two_epoch_seconds']
    remaining=budget['deadline_unix']-time.time()
    if remaining<estimate*1.10+45:
        pending=dict(experiment=identifier,status='pendiente_GPU',estimate_seconds=estimate,remaining_seconds=remaining,decision=decision)
        (session/f'pending_{identifier}.json').write_text(json.dumps(pending,indent=2),encoding='utf-8')
        print(json.dumps(pending),flush=True);return
    cfg.update(DEADLINE_UNIX=budget['deadline_unix']-30,EXPERIMENT_DECISION=decision)
    started=time.time();directory,summary=train(cfg)
    budget['completed'].append(dict(experiment=identifier,run=directory.name,seconds=time.time()-started,decision=decision,status=summary['status']))
    budget['elapsed_seconds']=time.time()-budget['start_unix']
    (session/'budget.json').write_text(json.dumps(budget,indent=2),encoding='utf-8')
    frame=comparison();names={e['run'] for e in budget['completed']}
    frame[frame.Run.isin(names)].to_csv(session/'comparison.csv',index=False)
    print(json.dumps(dict(run=directory.name,summary=summary,remaining_seconds=budget['deadline_unix']-time.time()),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run');parser.add_argument('--decision',default='Comparación preespecificada de una variable')
    args=parser.parse_args();session=project_path(args.session)
    if args.prepare:prepare(session)
    elif args.run:execute(session,args.run,args.decision)
    else:parser.error('Indicar --prepare o --run')


if __name__=='__main__':main()
