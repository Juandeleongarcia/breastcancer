"""Figuras y resúmenes F13: valores originales, cobertura explícita, sin smoothing."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from src.metrics import binary_metrics
from src.phase13_clinical import digest

TRAIN, VALIDATION = '#007E87', '#8752A1'


def unique_directory(parent, label):
    stamp = datetime.now(ZoneInfo('Europe/Madrid')).strftime('%Y%m%d_%H%M%S')
    name = f'RUN_{stamp}_{label[4:]}_{uuid4().hex[:12]}' if label.startswith('RUN_') else f'{label}_{stamp}_{uuid4().hex[:12]}'
    path = Path(parent)/name
    path.mkdir(parents=True,exist_ok=False)
    return path


def finish(fig,ax,path,title,ylabel,xlabel='Época'):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def history_plots(history, output, ylim=(0,1)):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(history)
    if frame.empty:return
    has_val = 'val_loss' in frame
    for metric,ylabel,title in [('loss','BCE media por corte','Pérdidas comparables · pesos de final de época'),
            ('accuracy','Accuracy por paciente','Accuracy por paciente'),('auc','ROC AUC por paciente','Discriminación por paciente')]:
        fig,ax = plt.subplots(figsize=(9,5))
        for prefix,label,color in [('train_eval','Train eval · sin augmentation',TRAIN),('val','Validación interna',VALIDATION)]:
            column = f'{prefix}_loss' if metric=='loss' else f'{prefix}_patient_{metric}'
            if column in frame:ax.plot(frame.epoch,frame[column],'o-',label=label,color=color)
        if metric!='loss':ax.set_ylim(0,1)
        if metric=='auc':ax.axhline(.5,color='#7B8794',ls=':',label='Referencia aleatoria')
        ax.set_xticks(frame.epoch);ax.legend(fontsize=9)
        finish(fig,ax,output/f'{metric}_curve.png',title,ylabel)
    extra = output/'internals';extra.mkdir(exist_ok=True)
    for column,label,color in [('train_eval_loss','Train eval',TRAIN),('val_loss','Validación',VALIDATION)]:
        if column not in frame:continue
        fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(frame.epoch,frame[column],'o-',color=color,label=label)
        ax.legend();finish(fig,ax,extra/(column+'_curve.png'),label+' · evaluación de final de época','BCE media por corte')
    if has_val:
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(frame.epoch,frame.train_eval_loss,'o-',label='Train eval',color=TRAIN)
        ax.plot(frame.epoch,frame.val_loss,'o-',label='Validación',color=VALIDATION)
        ax.set_ylim(*ylim);ax.legend()
        title=f'Pérdidas · eje limitado a {ylim[0]}–{ylim[1]} (ver loss_curve.png sin recorte)'
        finish(fig,ax,extra/'loss_curve_limited.png',title,'BCE media por corte')


def outcome_plots(predictions, output, threshold=.5, title='Validación por paciente'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=binary_metrics(predictions.label,predictions.probability,threshold)
    fig,ax=plt.subplots(figsize=(6,5.5))
    if predictions.label.nunique()==2:
        fpr,tpr,_=roc_curve(predictions.label,predictions.probability)
        ax.plot(fpr,tpr,color=VALIDATION,lw=2,label=f'ROC AUC = {metrics["auc"]:.4f}')
    ax.plot([0,1],[0,1],'--',color='#7B8794',label='Referencia aleatoria')
    ax.set(xlim=(0,1),ylim=(0,1.01));ax.legend(loc='lower right')
    finish(fig,ax,output/'roc_curve.png',title,'Sensibilidad','1 − especificidad')
    fig,ax=plt.subplots(figsize=(5.5,5))
    matrix=np.array(metrics['confusion_matrix'])
    ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=16)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['no pCR','pCR'],yticklabels=['no pCR','pCR'])
    finish(fig,ax,output/'confusion_matrix_patient.png',title+f' · umbral {threshold:g}','Etiqueta real','Predicción')
    return metrics


def mean_history(histories):
    records=[]
    for fold,frame in histories.items():
        for row in frame.itertuples():
            if np.isfinite([row.train_eval_loss,row.val_loss]).all():
                records.append(dict(fold=fold,epoch=row.epoch,train=row.train_eval_loss,val=row.val_loss))
    if not records:return pd.DataFrame()
    return pd.DataFrame(records).groupby('epoch').agg(train_mean=('train','mean'),train_std=('train','std'),
        val_mean=('val','mean'),val_std=('val','std'),n_folds=('fold','nunique')).reset_index()


def mean_plots(table, output):
    if table.empty:return
    fig,(ax,counts)=plt.subplots(2,1,figsize=(9,6.5),sharex=True,gridspec_kw={'height_ratios':[4,1]})
    for prefix,label,color in [('train','Train eval',TRAIN),('val','Validación',VALIDATION)]:
        ax.plot(table.epoch,table[prefix+'_mean'],'o-',label=label,color=color)
        ax.fill_between(table.epoch,table[prefix+'_mean']-table[prefix+'_std'],
                        table[prefix+'_mean']+table[prefix+'_std'],color=color,alpha=.15)
    ax.set(title='Pérdidas medias ± DE muestral · sin relleno tras early stopping',ylabel='BCE media por corte')
    ax.grid(alpha=.18);ax.legend()
    counts.bar(table.epoch,table.n_folds,color='#547181');counts.set(ylabel='Folds',xlabel='Época',ylim=(0,5.5))
    counts.set_xticks(table.epoch);counts.set_yticks([1,2,3,4,5]);fig.tight_layout()
    fig.savefig(Path(output)/'mean_loss_curve.png',dpi=160);plt.close(fig)


def recommend(rows, histories, config):
    if len(rows)!=5 or set(row['fold'] for row in rows)!=set(range(5)):
        raise ValueError('La recomendación exige cinco folds completos, no SMOKE_TEST')
    epochs=[int(row['best_epoch']) for row in sorted(rows,key=lambda r:r['fold'])]
    median=float(np.median(epochs));chosen=int(median)
    schedule=[]
    for epoch in range(1,chosen+1):
        values=[float(h.loc[h.epoch.eq(epoch),'learning_rate'].iloc[0])
                for h in histories.values() if h.epoch.eq(epoch).any()]
        schedule.append(dict(epoch=epoch,learning_rate=float(np.median(values)),n_folds=len(values)))
    return dict(schema_version=1,best_epochs=epochs,criterion='maximum_validation_patient_auc_per_fold_first_tie',
        recommended_epochs=chosen,median=median,mean=float(np.mean(epochs)),
        std_sample=float(np.std(epochs,ddof=1)),minimum=min(epochs),maximum=max(epochs),
        justification='Mediana de las mejores épocas de los cinco folds internos; robusta frente a extremos. No garantiza un óptimo ni usa test.',
        final_lr_schedule=schedule,
        final_scheduler_policy='fixed median of observed CV learning rates per epoch; no validation or test feedback',
        configuration=config,configuration_sha256=digest(config),external_test_used=False)


def validate_patient_predictions(frame, validation):
    expected=validation.groupby('patient_id',sort=True).agg(label=('pCR','first'),n_slices=('sample_id','size')).reset_index()
    if (frame.patient_id.duplicated().any() or set(frame.patient_id)!=set(expected.patient_id)
            or not {'label','probability','n_slices'}.issubset(frame)):
        raise ValueError('Cobertura de predicciones de paciente no válida')
    joined=frame.merge(expected,on='patient_id',validate='one_to_one',suffixes=('','_expected'))
    if not joined.label.eq(joined.label_expected).all() or not joined.n_slices.eq(joined.n_slices_expected).all():
        raise ValueError('Etiquetas o número de cortes no corresponden a la validación completa')
    binary_metrics(frame.label,frame.probability)
    return frame


def write_json(path,value):
    """JSON determinista; la sustitución atómica la proporciona la utilidad original."""
    from src.cross_validation import save_json
    save_json(Path(path),value)
