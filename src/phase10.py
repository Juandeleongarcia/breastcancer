"""Control manual F10: campañas persistentes y un proceso nuevo por fold GPU."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import signal
import statistics
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
SMALL='SPATIAL_SMALL'
DROPOUT='F8_SMALL_DROPOUT'
LOW_LR='F8_SMALL_LR'
MODELS=[SMALL,DROPOUT,LOW_LR]
EXISTING='models/CV9_20261028_183759_SPATIAL_SMALL_531c826715d2'


def path(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def read(value):
    return json.loads(path(value).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save(filename,value):
    filename=Path(filename);filename.parent.mkdir(parents=True,exist_ok=True)
    temporary=filename.with_name(filename.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,filename)


def templates():
    """Reusar F8 íntegro salvo identificación y campos de ajuste CV de F9."""
    base=read('config/phase9/SPATIAL_SMALL.json')
    original=read('config/phase7/SPATIAL_SMALL.json')
    result={SMALL:base}
    preprocessing=['STATS_PATH','STATS_PROVENANCE','STATS_PREPROCESSING','CHANNEL_P01',
                   'CHANNEL_P99','CHANNEL_MEANS','CHANNEL_STDS',
                   'EXPECTED_TRAIN_PATIENTS','EXPECTED_VAL_PATIENTS']
    for name,variable,value in [(DROPOUT,'DROPOUT',.5),(LOW_LR,'LEARNING_RATE',.0001)]:
        cfg=read(f'config/phase8/{name}.json')
        changed={k for k in set(cfg)|set(original) if cfg.get(k)!=original.get(k)}-{'RUN_NAME','COMPARISON_PARENT'}
        if changed!={variable} or cfg[variable]!=value:
            raise ValueError('F8 cambió respecto a su control: '+name+' '+str(changed))
        inherited=dict(cfg)
        for key in preprocessing:cfg[key]=base[key]
        cfg.update(INTERPRETATION='phase10_official_cv_from_'+name,COMPARISON_PARENT='F9_SPATIAL_SMALL')
        ignored=set(preprocessing)|{'INTERPRETATION','COMPARISON_PARENT'}
        if any(cfg[k]!=inherited[k] for k in inherited if k not in ignored):
            raise ValueError('Se alteraron hiperparámetros F8')
        result[name]=cfg
    for cfg in result.values():
        if (cfg['CONV_CHANNELS']!=[16,32,64] or cfg['CONVS_PER_BLOCK']!=2
                or cfg['INPUT_CHANNELS']!=3 or cfg['EPOCHS']!=20 or cfg['PATIENCE']!=8
                or cfg['PRIMARY_METRIC']!='patient_auc' or not cfg['EVALUATE_TRAIN']):
            raise ValueError('Configuración fuera del protocolo de clase')
    return result


def active_trainers(proc=Path('/proc'), own=None):
    """Solo lectura; fallar si /proc no permite comprobar procesos. Nunca matar PIDs."""
    own=os.getpid() if own is None else own
    if not proc.is_dir():raise RuntimeError('No se puede comprobar /proc; no iniciar GPU')
    found=[]
    for item in proc.iterdir():
        if not item.name.isdecimal() or int(item.name)==own:continue
        try:
            args=(item/'cmdline').read_bytes().decode(errors='replace').split('\0')
            status=(item/'stat').read_text().rpartition(')')[2].split()
            if status and status[0]=='Z':continue
        except FileNotFoundError:continue
        except PermissionError as error:raise RuntimeError('No se puede verificar otro proceso; esperar al fin del entrenamiento anterior') from error
        modules={'src.cross_validation','src.train','src.experiment','src.batch_train'}
        recognized=any(a=='-m' and i+1<len(args) and args[i+1] in modules for i,a in enumerate(args))
        recognized|=any(Path(a).name in {'cross_validation.py','train.py','batch_train.py'} for a in args if a)
        if recognized:found.append(int(item.name))
    return found


@contextmanager
def gpu_lock():
    import fcntl  # Linux, nunca se importa al validar configs/tests en Windows.
    lock=ROOT/'models/.phase10_gpu.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Otro controlador F10 ya está activo') from error
        busy=active_trainers()
        if busy:raise RuntimeError(f'Entrenamiento existente activo, PID {busy}; esperar, no lanzar otro')
        yield


def variant_campaign(name,state_dir):
    """Adoptar la única campaña propia tras una interrupción; no crear duplicados."""
    from src.cross_validation import prepare_campaign,checked_campaign
    wanted=templates()[name]
    state_dir=path(state_dir);registry=state_dir/'campaigns.json'
    routes=read(registry) if registry.exists() else {}
    if name in routes:
        root=path(routes[name])
    else:
        parent=state_dir/'campaigns'/name
        candidates=sorted(parent.glob('CV9_*/campaign.json'))
        if len(candidates)>1:raise ValueError('Varias campañas candidatas; revisar sin crear otra: '+str(parent))
        root=candidates[0].parent if candidates else prepare_campaign(SMALL,output_root=parent,template=wanted)
    _,state,_=checked_campaign(root)
    if state['diagnostic'] or digest(read(root/'template.json'))!=digest(wanted):
        raise ValueError('Campaña candidata/configuración incompatible: '+str(root))
    if state['model']!=name:
        if state['model']!=SMALL or any(e['status']!='prepared' for e in state['folds'].values()):
            raise ValueError('No se cambia la identidad de una campaña ya entrenada')
        # Solo metadatos de la campaña NUEVA, antes de ajustar/entrenar ningún fold.
        state.update(model=name,base_architecture=SMALL,source_config=f'config/phase8/{name}.json')
        save(root/'campaign.json',state)
        # La Fase 9 permanece intacta; conservar también la fuente del adaptador.
        (root/'phase10_adapter.py').write_bytes(Path(__file__).read_bytes())
    routes[name]=str(root.resolve());save(registry,routes)
    return root


def checkpoint_candidates(cfg):
    """Ignorar archivos corruptos/incompletos y elegir el último límite compatible."""
    import torch
    from src.checkpointing import load_resume
    from src.resume_validation import validate_resume_environment
    candidates=[];rejected=[]
    for root in path(cfg['RESULTS_DIR']).glob('RUN_*'):
        for folder in [root,root/'internals',root/'checkpoints']:
            files=set(folder.glob('epoch_*.pt'))|set(folder.glob('recovery_epoch_start*.pt'))
            for filename in files:
                try:
                    validate_resume_environment(filename)
                    payload=load_resume(filename,cfg)
                    candidates.append((payload['epoch'],filename.stat().st_mtime_ns,filename))
                except Exception as error:
                    rejected.append(dict(path=str(filename),reason=str(error)))
    return (max(candidates)[2] if candidates else None),rejected


def complete_details(cfg):
    import pandas as pd
    import torch
    from src.batch_train import completed_run
    from src.run_layout import technical_file
    root=completed_run(cfg)
    if root is None:return None
    summary=read(technical_file(root,'summary.json'));history=pd.read_csv(root/'history.csv')
    n=int(summary['epochs_completed']);best=int(summary['best_epoch'])
    if (not n or list(history.epoch)!=list(range(1,n+1)) or best not in set(history.epoch)
            or (n<cfg['EPOCHS'] and not summary.get('stopped_early'))
            or not summary.get('validation_complete') or summary.get('diagnostic_partial')):
        raise ValueError('Resumen incompleto/inconsistente: '+str(root))
    for name,epoch in [('best_epoch.pt',best),(f'epoch_{n:03d}.pt',n)]:
        checkpoint=torch.load(root/name,map_location='cpu',weights_only=True)
        if checkpoint.get('epoch')!=epoch or not checkpoint.get('resume_supported'):
            raise ValueError('Checkpoint completo inválido: '+str(root/name))
    for name in ['config.py','loss_curve.png','loss_comparison.png','accuracy_curve.png','auc_curve.png','roc_curve.png']:
        if not (root/name).is_file():raise ValueError('Falta resultado: '+str(root/name))
    pred=pd.read_csv(technical_file(root,'predictions_patient.csv'))
    expected=pd.read_csv(path(cfg['VAL_CSV']))[['patient_id','pCR']].drop_duplicates()
    joined=pred.merge(expected,on='patient_id',validate='one_to_one')
    if set(pred.patient_id)!=set(expected.patient_id) or not joined.label.eq(joined.pCR).all():
        raise ValueError('Predicciones de validación no cubren el fold')
    row=history[history.epoch.eq(best)].iloc[0]
    metrics=summary['validation']['patient']
    return dict(status='COMPLETO',run=str(root),best_checkpoint=str(root/'best_epoch.pt'),best_epoch=best,
                epochs=n,auc=metrics['auc'],accuracy=metrics['accuracy'],
                balanced_accuracy=(metrics['recall']+metrics['specificity'])/2,
                train_eval_loss=float(row.train_eval_loss),val_loss=summary['validation']['loss'])


def inspect_campaign(root):
    from src.cross_validation import checked_campaign
    root,state,_=checked_campaign(root)
    if state['diagnostic']:raise ValueError('No se usa una campaña smoke para la CV final')
    rows=[]
    for fold in range(5):
        entry=state['folds'][str(fold)];folder=root/f'fold_{fold}'
        cfg=read(folder/entry['config_file']) if entry.get('config_file') else None
        complete=complete_details(cfg) if cfg else None
        if complete:rows.append(dict(fold=fold,**complete));continue
        if entry['status']=='complete':raise ValueError(f'Fold {fold} marcado completo pero sin evidencia íntegra; no se repetirá')
        existing=list((folder/'runs').glob('RUN_*'))
        checkpoint,rejected=checkpoint_candidates(cfg) if cfg and existing else (None,[])
        rows.append(dict(fold=fold,status='RECUPERABLE' if checkpoint else 'SIN_CHECKPOINT' if existing else 'PENDIENTE',
                         checkpoint=str(checkpoint) if checkpoint else None,rejected_checkpoints=rejected,error=entry.get('error')))
    return root,state,rows


def validate_small(root):
    from src.cross_validation import checked_campaign
    root,state,_=checked_campaign(root)
    cfg=read(root/'template.json')
    if (state['diagnostic'] or state['model']!=SMALL or cfg['CONV_CHANNELS']!=[16,32,64]
            or cfg['CONVS_PER_BLOCK']!=2 or cfg['EPOCHS']>20
            or cfg['PRIMARY_METRIC']!='patient_auc'):
        raise ValueError('La ruta no corresponde a la campaña SMALL oficial esperada')
    print('Configuración REAL de SMALL:',json.dumps({k:cfg[k] for k in
          ['LEARNING_RATE','DROPOUT','OPTIMIZER','WEIGHT_DECAY','BATCH_SIZE','SEED','EPOCHS','PATIENCE']}),flush=True)
    return root


class Executor:
    def __init__(self):self.child=None;self.stop=False
    def interrupt(self,signum,frame):
        if self.stop:return
        self.stop=True
        if self.child is None:raise KeyboardInterrupt()
        if signum==signal.SIGTERM:self.child.send_signal(signal.SIGINT)
        # Ctrl+C llega también al hijo del mismo grupo; dejar que guarde recovery.
    def __call__(self,args):
        print('Ejecutando:',subprocess.list2cmdline(args),flush=True)
        self.child=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                    text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in self.child.stdout:print(line,end='',flush=True)
            return self.child.wait()
        finally:self.child=None


def run_manual(root,folds=None,restart_folds=(),execute=None):
    from src.cross_validation import checked_campaign,fit_job
    execute=execute or Executor()
    for fold in (range(5) if folds is None else folds):
        root,state,rows=inspect_campaign(root);row=rows[fold]
        if row['status']=='COMPLETO':
            entry=state['folds'][str(fold)]
            if entry['status']!='complete' or entry.get('last_run')!=row['run']:
                entry.update(status='complete',last_run=row['run'])
                save(root/'campaign.json',state)
            print(f'Fold {fold}: completo, se omite',flush=True);continue
        busy=active_trainers()
        if busy:raise RuntimeError(f'Hay otro entrenamiento activo: {busy}')
        cfg=fit_job(root,fold)
        checkpoint,rejected=checkpoint_candidates(cfg)
        existing=list(path(cfg['RESULTS_DIR']).glob('RUN_*'))
        unused_checkpoint=checkpoint if existing and fold in restart_folds else None
        if unused_checkpoint:checkpoint=None  # Reinicio únicamente por petición explícita.
        if existing and checkpoint is None and fold not in restart_folds:
            raise ValueError(f'Fold {fold}: sin checkpoint compatible. Revisar logs y, solo si procede, usar --restart-fold {fold}; se conservarán los RUNs')
        args=[sys.executable,'-u','-m','src.train','--device','cuda']
        if checkpoint:args+=['--resume',str(checkpoint),'--epochs',str(cfg['EPOCHS'])]
        else:args+=['--config',str(root/f'fold_{fold}'/read(root/'campaign.json')['folds'][str(fold)]['config_file'])]
        root,state,_=checked_campaign(root);entry=state['folds'][str(fold)]
        busy=active_trainers()  # Repetir tras un posible ajuste largo de estadísticas.
        if busy:raise RuntimeError(f'Se detectó otro entrenamiento antes de lanzar GPU: {busy}')
        entry.setdefault('phase10_attempts',[]).append(dict(time=datetime.now(timezone.utc).isoformat(),
            mode='resume' if checkpoint else 'restart_preserving_old_runs' if existing else 'fresh',
            checkpoint=str(checkpoint) if checkpoint else None,
            compatible_checkpoint_not_used=str(unused_checkpoint) if unused_checkpoint else None,
            previous_error=entry.get('error'),rejected_checkpoints=rejected))
        entry['status']='running';save(root/'campaign.json',state)
        rc=execute(args)
        complete=complete_details(cfg)
        root,state,_=checked_campaign(root);entry=state['folds'][str(fold)]
        if complete:
            entry.update(status='complete',last_run=complete['run'])
        else:
            created=sorted(path(cfg['RESULTS_DIR']).glob('RUN_*'),key=lambda p:p.stat().st_mtime_ns)
            entry.update(status='interrupted' if getattr(execute,'stop',False) else 'error',
                         error=f'Proceso nuevo finalizado con código {rc}; fold incompleto. Consultar log F10')
            if created:entry['last_run']=str(created[-1])
        save(root/'campaign.json',state)
        if getattr(execute,'stop',False):raise KeyboardInterrupt()
        if rc or not complete or getattr(execute,'stop',False):
            raise RuntimeError(f'Fold {fold} detenido; no se reintenta ni se lanza otro fold en este proceso')
    from src.cross_validation import summarize_campaign
    return summarize_campaign(root)


def comparison(state_dir,small):
    state_dir=path(state_dir);routes=read(state_dir/'campaigns.json') if (state_dir/'campaigns.json').exists() else {}
    roots={SMALL:str(path(small)),**routes};rows=[];details=[]
    for model in MODELS:
        row=dict(model=model,campaign=roots.get(model),status='INCOMPLETA',complete_folds=0)
        if roots.get(model) and path(roots[model]).is_dir():
            _,_,folds=inspect_campaign(roots[model]);done=[r for r in folds if r['status']=='COMPLETO']
            row['complete_folds']=len(done)
            details.extend(dict(model=model,**r) for r in folds)
            if len(done)==5:
                from src.cross_validation import summarize_campaign
                summary=summarize_campaign(roots[model])
                if summary['complete_folds']!=5:raise ValueError('Estado de campaña no reconciliado; ejecutar su comando manual')
                row.update(status='COMPLETA',auc_mean=summary['auc_mean'],auc_std_sample=summary['auc_std_sample'],
                           oof_auc=summary['oof_patient_auc'],accuracy_mean=statistics.mean(r['accuracy'] for r in done),
                           balanced_accuracy_mean=statistics.mean(r['balanced_accuracy'] for r in done),
                           train_eval_loss_mean=statistics.mean(r['train_eval_loss'] for r in done),
                           val_loss_mean=statistics.mean(r['val_loss'] for r in done))
        rows.append(row)
    output=state_dir/'reports';output.mkdir(parents=True,exist_ok=True)
    token=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:12]
    for name,data in [('comparison',rows),('folds',details)]:
        if not data:continue
        fields=list(dict.fromkeys(k for row in data for k in row))
        with (output/f'{name}_{token}.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    save(output/f'report_{token}.json',dict(models=rows,folds=details))
    print(json.dumps(rows,indent=2));print('Informes:',output)
    return rows


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,text):self.console.write(text);self.log.write(text);self.flush();return len(text)
    def flush(self):self.console.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.console,name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['configs','status','small','dropout','lr','compare'])
    parser.add_argument('--campaign',default=EXISTING)
    parser.add_argument('--state-dir',default='models/phase10')
    parser.add_argument('--folds',nargs='+',type=int,choices=range(5))
    parser.add_argument('--restart-fold',nargs='+',type=int,choices=range(5),default=[])
    args=parser.parse_args()
    if args.action=='configs':
        values=templates()
        print(json.dumps({name:{k:cfg[k] for k in ['CONV_CHANNELS','LEARNING_RATE','DROPOUT','OPTIMIZER','WEIGHT_DECAY','BATCH_SIZE','SEED','PATIENCE','EPOCHS','USE_AUGMENTATION','USE_WEIGHTED_LOSS','INPUT_MODE','USE_CLIPPING','USE_STANDARDIZATION']} for name,cfg in values.items()},indent=2));return
    if args.action=='status':
        _,_,rows=inspect_campaign(args.campaign);print(json.dumps(rows,indent=2));return
    if not sys.platform.startswith('linux'):parser.error('Ejecutar estos comandos de GPU en Linux; configs/tests sí funcionan en Windows')
    model={'small':SMALL,'dropout':DROPOUT,'lr':LOW_LR}.get(args.action,'comparison')
    logdir=path(args.state_dir)/'logs';logdir.mkdir(parents=True,exist_ok=True)
    original=sys.stdout
    with gpu_lock(),(logdir/(model+'.log')).open('a',encoding='utf-8',buffering=1) as logfile:
        sys.stdout=Tee(original,logfile)
        try:
            print('\nF10',datetime.now(timezone.utc).isoformat(),args.action,flush=True)
            if args.action=='compare':comparison(args.state_dir,args.campaign);return
            templates()
            executor=Executor();signal.signal(signal.SIGINT,executor.interrupt);signal.signal(signal.SIGTERM,executor.interrupt)
            code='from src.check_suite import require_rocm; from src.batch_train import check_device; require_rocm(); check_device("cuda")'
            if executor([sys.executable,'-u','-c',code]):raise RuntimeError('Preflight ROCm falló; no se inicia entrenamiento')
            root=path(args.campaign) if model==SMALL else variant_campaign(model,args.state_dir)
            if not root.is_dir():raise FileNotFoundError('Campaña de clase ausente: '+str(root))
            if model==SMALL:validate_small(root)
            result=run_manual(root,args.folds,args.restart_fold,executor)
            print(json.dumps(result,indent=2))
            comparison(args.state_dir,args.campaign)
        except (Exception,KeyboardInterrupt) as error:
            print('DETENIDO / INCOMPLETO:',str(error),flush=True)
            raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 1)
        finally:sys.stdout=original


if __name__=='__main__':main()
