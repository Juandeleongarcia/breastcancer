"""Motor F13 independiente: CPU/ROCm, evaluaciones comparables y recuperación por época."""
from pathlib import Path
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.checkpointing import atomic_save, capture_rng, restore_rng
from src.experiment import set_seed, select_device, create_optimizer, create_criterion, environment_info
from src.metrics import evaluate_predictions, analyze_probabilities
from src.phase13_clinical import ClinicalTransform, digest
from src.phase13_data import MultimodalDataset
from src.phase13_model import SpatialSmallMultimodal, ARCHITECTURE
from src.phase13_reports import history_plots, outcome_plots, write_json


def inference_contract(cfg, clinical):
    keys=['INPUT_CHANNELS','IMAGE_SIZE','INPUT_MODE','USE_CLIPPING','CHANNEL_P01','CHANNEL_P99',
          'USE_STANDARDIZATION','CHANNEL_MEANS','CHANNEL_STDS','STATS_PROVENANCE','STATS_PREPROCESSING']
    return dict(schema_version=13,architecture=ARCHITECTURE,clinical_transform=clinical,
        image_preprocessing={k:cfg[k] for k in keys},image_shape=[3,cfg['IMAGE_SIZE'],cfg['IMAGE_SIZE']],
        clinical_input_columns=['HR_HER2_STATUS','age'],clinical_encoded_columns=clinical['columns'],
        threshold=cfg['CLASSIFICATION_THRESHOLD'],aggregation='mean_of_slice_sigmoid_probabilities',
        model_inputs=['image','clinical'],model_output='single_pCR_logit_per_slice',
        selection_status='experimental_not_selected_for_Streamlit',scope=cfg['TRAINING_SCOPE'])


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=.5):
    training=optimizer is not None
    model.train(training)
    rows=[];total=0.;count=0;last=time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            images=batch['image'].to(device);clinical=batch['clinical'].to(device)
            labels=batch['label'].to(device).reshape(-1)
            if not torch.isfinite(images).all() or not torch.isfinite(clinical).all():
                raise FloatingPointError('Entrada multimodal no finita')
            if training:optimizer.zero_grad(set_to_none=True)
            logits=model(images,clinical).squeeze(1)
            loss=criterion(logits,labels)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError('Loss/logits no finitos')
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Gradientes no finitos')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Pesos no finitos')
            rows.extend(dict(sample_id=str(sid),patient_id=str(pid),label=int(y),probability=float(p))
                for sid,pid,y,p in zip(batch['sample_id'],batch['patient_id'],labels.detach().cpu().tolist(),
                                      torch.sigmoid(logits.detach()).cpu().tolist()))
            count+=len(labels);total+=float(loss.detach())*len(labels)
            if time.perf_counter()-last>=30:
                print(f'{"Train online" if training else "Eval"}: {count}/{len(loader.dataset)} cortes; BCE={total/count:.5f}',flush=True)
                last=time.perf_counter()
    if count!=len(loader.dataset):raise ValueError('No se permite una pasada parcial en F13')
    result=evaluate_predictions(rows,threshold);result['loss']=total/count
    return result


def evaluate(model, dataset, criterion, device, cfg, generator):
    state=capture_rng(generator)
    modes={module:module.training for module in model.modules()}
    try:
        loader=DataLoader(dataset,batch_size=cfg['BATCH_SIZE'],shuffle=False,num_workers=cfg['NUM_WORKERS'],
            pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(cfg['SEED']))
        with torch.inference_mode():
            return run_epoch(model,loader,criterion,device,threshold=cfg['CLASSIFICATION_THRESHOLD'])
    finally:
        restore_rng(state,generator,device)
        for module,mode in modes.items():module.training=mode


def scalar(result):
    return dict(loss=result['loss'],patient=result['patient'],slice=result['slice'])


def history_row(epoch, online, evaluated, validation, lr, seconds, smoke):
    row=dict(epoch=epoch,learning_rate=lr,epoch_time_seconds=seconds,SMOKE_TEST=smoke,
             train_metric_protocol='online_changing_weights_dropout',
             train_eval_metric_protocol='end_of_epoch_eval_no_augmentation',
             validation_metric_protocol='end_of_epoch_eval_no_augmentation')
    for prefix,result in [('train',online),('train_eval',evaluated),('val',validation)]:
        if result is None:continue
        row[prefix+'_loss']=result['loss']
        for level in ['patient','slice']:
            for key,value in result[level].items():
                row[f'{prefix}_{level}_{key}']=json.dumps(value) if key=='confusion_matrix' else value
    if validation:row['loss_gap']=validation['loss']-evaluated['loss']
    return row


def compatible_checkpoint(path, contract, cfg, device, allow_device_change=False):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (payload.get('schema_version')!=13 or payload.get('architecture')!=ARCHITECTURE
            or not payload.get('resume_supported') or payload.get('epoch_incomplete')
            or payload.get('training_contract')!=contract):
        raise ValueError('Checkpoint F13 incompatible con fuentes, clínica, particiones o configuración')
    if payload['device']!=str(device) and not allow_device_change:
        raise ValueError('Cambio de backend: indicar --allow-device-change; la trayectoria numérica puede variar')
    if payload['epoch']>=cfg['EPOCHS'] or (cfg['TRAINING_SCOPE']=='cross_validation'
            and payload['waiting']>=cfg['PATIENCE']):
        raise ValueError('El entrenamiento ya finalizó; no se reinician épocas ni early stopping')
    if payload['clinical_transform']!=contract['clinical_transform']:
        raise ValueError('Transformador clínico incompatible')
    if payload['best_epoch']:
        folder=Path(path).parent
        if folder.name=='internals':folder=folder.parent
        candidates=[folder/'best_epoch.pt',folder/f'epoch_{payload["best_epoch"]:03d}.pt']
        valid=False
        for candidate in candidates:
            if candidate.is_file():
                chosen=torch.load(candidate,map_location='cpu',weights_only=True)
                if (chosen.get('epoch')==payload['best_epoch'] and chosen.get('resume_supported')
                        and chosen.get('training_contract')==contract):
                    valid=True;break
        if not valid:raise ValueError('Falta el checkpoint completo de la mejor época heredada; restaurar sus artefactos')
    return payload


def train_job(cfg, clinical_state, patients, contract, directory, resume_from=None,
              allow_device_change=False, final_schedule=None, stop_after_epoch=None):
    """stop_after_epoch es una sonda de tests; no existe en la CLI ni limita campañas."""
    directory=Path(directory);details=directory/'internals';details.mkdir(exist_ok=True)
    final=cfg['TRAINING_SCOPE']=='final_development'
    if final and (not final_schedule or len(final_schedule)!=cfg['EPOCHS']):
        raise ValueError('Entrenamiento final requiere calendario fijado por CV')
    set_seed(cfg['SEED']);device=select_device(cfg)
    transform=ClinicalTransform(clinical_state)
    tr=MultimodalDataset(cfg['TRAIN_CSV'],cfg['DATA_DIR'],cfg,patients,transform,augment=True)
    te=MultimodalDataset(cfg['TRAIN_CSV'],cfg['DATA_DIR'],cfg,patients,transform)
    va=None if final else MultimodalDataset(cfg['VAL_CSV'],cfg['DATA_DIR'],cfg,patients,transform)
    if (not tr.df.split.eq('train').all() or not tr.df.fold.isin(range(5)).all()
            or (va is not None and (set(tr.df.patient_id)&set(va.df.patient_id)
                or not va.df.fold.eq(cfg['CV_FOLD']).all() or tr.df.fold.eq(cfg['CV_FOLD']).any()))):
        raise ValueError('Particiones de entrenamiento no autorizadas')
    generator=torch.Generator().manual_seed(cfg['SEED'])
    loader=DataLoader(tr,batch_size=cfg['BATCH_SIZE'],shuffle=True,generator=generator,
                      num_workers=cfg['NUM_WORKERS'],pin_memory=device.type=='cuda')
    model=SpatialSmallMultimodal(cfg,transform.dimension).to(device)
    criterion=create_criterion(cfg,device);optimizer=create_optimizer(model,cfg)
    scheduler=None if final or cfg['SCHEDULER']=='none' else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='max',factor=cfg['SCHEDULER_FACTOR'],patience=cfg['SCHEDULER_PATIENCE'],
        min_lr=cfg['SCHEDULER_MIN_LR'],threshold=0.)
    history=[];best=-math.inf;best_epoch=0;waiting=0;best_weights=None;best_rows=[];best_loss=None
    start_epoch=1;prior_seconds=0.;started=time.perf_counter()
    if resume_from:
        saved=compatible_checkpoint(resume_from,contract,cfg,device,allow_device_change)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        if scheduler:scheduler.load_state_dict(saved['scheduler_state_dict'])
        history=saved['history'];best=saved['best_auc'];best_epoch=saved['best_epoch'];waiting=saved['waiting']
        best_weights=saved['best_model_state_dict'];best_rows=saved['best_rows'];best_loss=saved['best_loss']
        start_epoch=saved['epoch']+1;prior_seconds=saved['elapsed_seconds']
        restore_rng(saved['rng_state'],generator,device)
        write_json(details/'resume.json',dict(source=str(resume_from),completed_epoch=saved['epoch'],
            device_change=saved['device']!=str(device),policy='new_RUN_replay_interrupted_epoch_from_its_start'))
    interface=inference_contract(cfg,clinical_state)
    write_json(details/'config.json',cfg);write_json(details/'clinical_transform.json',clinical_state)
    write_json(details/'training_contract.json',contract);write_json(details/'inference_contract.json',interface)
    write_json(details/'environment.json',environment_info(device))
    (directory/'config.py').write_text('# Snapshot literal de F13. No ejecutar para cargar la configuración.\n'+
        '\n'.join(f'{key} = {value!r}' for key,value in cfg.items()),encoding='utf-8')
    # Un único ZIP de fuentes, sin datos ni rutas de pacientes.
    from zipfile import ZipFile, ZIP_DEFLATED
    from src.configuration import project_path
    with ZipFile(details/'sources.zip','x',compression=ZIP_DEFLATED) as archive:
        for name in contract['implementation']:archive.write(project_path(name),name)
    if history:pd.DataFrame(history).to_csv(directory/'history.csv',index=False)

    def snapshot(epoch):
        return dict(schema_version=13,architecture=ARCHITECTURE,resume_supported=True,epoch_incomplete=False,
            epoch=epoch,device=str(device),config=cfg,clinical_transform=clinical_state,
            training_contract=contract,inference_contract=interface,model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            rng_state=capture_rng(generator),history=history,best_auc=best,best_epoch=best_epoch,waiting=waiting,
            best_model_state_dict=best_weights,best_rows=best_rows,best_loss=best_loss,
            elapsed_seconds=prior_seconds+time.perf_counter()-started)

    if best_epoch:
        source=Path(resume_from).parent
        if source.name=='internals':source=source.parent
        candidates=[source/'best_epoch.pt',source/f'epoch_{best_epoch:03d}.pt']
        for candidate in candidates:
            if candidate.is_file():
                chosen=torch.load(candidate,map_location='cpu',weights_only=True)
                if chosen.get('epoch')==best_epoch and chosen.get('training_contract')==contract:
                    shutil.copy2(candidate,directory/'best_epoch.pt');break
    print(f'RUN F13: {directory}\nDispositivo: {device}; SMOKE_TEST={cfg["SMOKE_TEST"]}; train={len(tr)} cortes',flush=True)
    status='interrupted'
    for epoch in range(start_epoch,cfg['EPOCHS']+1):
        if final:
            for group in optimizer.param_groups:group['lr']=final_schedule[epoch-1]['learning_rate']
        atomic_save(snapshot(epoch-1),details/'recovery_epoch_start.pt')
        began=time.perf_counter();lr=optimizer.param_groups[0]['lr']
        online=run_epoch(model,loader,criterion,device,optimizer,cfg['CLASSIFICATION_THRESHOLD'])
        evaluated=evaluate(model,te,criterion,device,cfg,generator)
        validation=evaluate(model,va,criterion,device,cfg,generator) if va is not None else None
        row=history_row(epoch,online,evaluated,validation,lr,time.perf_counter()-began,cfg['SMOKE_TEST'])
        improved=False
        if validation:
            score=validation['patient']['auc']
            if not math.isfinite(score):raise ValueError('AUC paciente no finito en validación completa')
            improved=score>best
            if improved:
                best=score;best_epoch=epoch;waiting=0
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_rows=validation['slice_predictions'][['sample_id','patient_id','label','probability']].to_dict('records')
                best_loss=validation['loss']
            else:waiting+=1
            if scheduler:scheduler.step(score)
        row['next_learning_rate']=optimizer.param_groups[0]['lr'];history.append(row)
        checkpoint=snapshot(epoch)
        atomic_save(checkpoint,directory/f'epoch_{epoch:03d}.pt')
        if improved:atomic_save(checkpoint,directory/'best_epoch.pt')
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        write_json(details/'epoch_metrics.json',history)
        history_plots(history,directory,tuple(cfg['LOSS_PLOT_YLIM']))
        print(f'Época {epoch}: train_eval={evaluated["loss"]:.5f}; '+
            (f'val={validation["loss"]:.5f}; AUC paciente={score:.5f}' if validation else 'final sin validación/test'),flush=True)
        if epoch>=cfg['EPOCHS'] or (not final and waiting>=cfg['PATIENCE']):
            status='complete';break
        if stop_after_epoch is not None and epoch>=stop_after_epoch:break
    summary=dict(status=status,SMOKE_TEST=cfg['SMOKE_TEST'],diagnostic_partial=cfg['SMOKE_TEST'],
        training_scope=cfg['TRAINING_SCOPE'],epochs_completed=len(history),requested_epochs=cfg['EPOCHS'],
        best_epoch=best_epoch if not final else None,best_patient_auc=best if not final and best_epoch else None,
        stopped_early=bool(not final and waiting>=cfg['PATIENCE']),validation_complete=bool(not final and best_epoch),
        train_patients=int(tr.df.patient_id.nunique()),validation_patients=int(va.df.patient_id.nunique()) if va else 0,
        elapsed_seconds=prior_seconds+time.perf_counter()-started,
        train_metric_note='train_loss online; train_eval y val con pesos finales, sin augmentation')
    if best_epoch:
        result=evaluate_predictions(best_rows,cfg['CLASSIFICATION_THRESHOLD']);result['loss']=best_loss
        summary['validation']=scalar(result)
        best_row=next(r for r in history if r['epoch']==best_epoch)
        summary['best_train_eval_loss']=best_row['train_eval_loss']
        summary['generalization_analysis']=dict(best_loss_gap=best_row['loss_gap'],
            last_loss_gap=history[-1]['loss_gap'],
            delta_train_eval_loss=history[-1]['train_eval_loss']-history[0]['train_eval_loss'],
            delta_validation_loss=history[-1]['val_loss']-history[0]['val_loss'],
            overfitting_signal=bool(len(history)>1 and history[-1]['train_eval_loss']<history[0]['train_eval_loss']
                                  and history[-1]['val_loss']>history[0]['val_loss']),
            predicted_pcr_fraction=float(result['patient_predictions'].prediction.mean()),
            note='Descripción de estas épocas; no demuestra superioridad ni calibración clínica')
        result['patient_predictions'].to_csv(details/'predictions_patient.csv',index=False)
        result['slice_predictions'].to_csv(details/'predictions_slice.csv',index=False)
        outcome_plots(result['patient_predictions'],directory,cfg['CLASSIFICATION_THRESHOLD'],'Validación por paciente · mejor AUC')
        # Mantener solo las cuatro curvas principales en la raíz del RUN.
        (directory/'confusion_matrix_patient.png').replace(details/'confusion_matrix_patient.png')
        analyze_probabilities(result,details,details)
    if final and status=='complete':
        shutil.copy2(directory/f'epoch_{cfg["EPOCHS"]:03d}.pt',directory/'final_epoch.pt')
    write_json(details/'summary.json',summary)
    return summary
