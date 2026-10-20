"""Memorización de pacientes de train: diagnóstico técnico, nunca validación clínica."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.batch_train import resolve_job
from src.dataset import BreastDCEDataset
from src.experiment import set_seed, select_device, select_patients, create_run_directory, environment_info
from src.metrics import evaluate_predictions, save_evaluation_artifacts
from src.model import BreastCancerCNN


def rank_info(matrix):
    matrix = matrix.float().flatten(1)
    centered = matrix - matrix.mean(0)
    singular = torch.linalg.svdvals(centered)
    weights = singular.square() / singular.square().sum().clamp_min(1e-20)
    effective = torch.exp(-(weights * weights.clamp_min(1e-20).log()).sum())
    return dict(dimensions=matrix.shape[1], effective_rank=float(effective),
                zero_variance_fraction=float((matrix.std(0) < 1e-6).float().mean()),
                mean_feature_std=float(matrix.std(0).mean()))


def probe(model, images, labels, rows, batch_size=16, batch_stats=False):
    saved = copy.deepcopy(model.state_dict()) if batch_stats else None
    model.train(batch_stats)
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.eval()
    probabilities, features, spatial = [], [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start:start+batch_size]
            maps = model.features[:-1](x)
            emb = model.features[-1](maps)
            logits = model.classifier(emb).squeeze(1)
            probabilities.extend(logits.sigmoid().tolist())
            features.append(emb.flatten(1))
            spatial.append(torch.nn.functional.adaptive_avg_pool2d(maps, (4,4)).flatten(1))
    if saved is not None:
        model.load_state_dict(saved)
    predictions = [dict(row, probability=float(p)) for row, p in zip(rows, probabilities)]
    result = evaluate_predictions(predictions)
    result['loss'] = float(nn.functional.binary_cross_entropy(torch.tensor(probabilities).clamp(1e-7, 1-1e-7), labels))
    result['features'] = rank_info(torch.cat(features))
    result['spatial_features'] = rank_info(torch.cat(spatial))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--max-seconds', type=float, default=240)
    parser.add_argument('--no-bn', action='store_true')
    parser.add_argument('--spatial', action='store_true')
    args = parser.parse_args()
    session = Path(args.session)
    cfg = resolve_job('config/phase4/today/01_NO_WEIGHT_PC.json', 'cpu')
    cfg.update(RUN_NAME='F5_DIAG_MEMORIZE_NO_BN' if args.no_bn else 'F5_DIAG_MEMORIZE_BASELINE',
               USE_AUGMENTATION=False, USE_BATCH_NORM=not args.no_bn, EPOCHS=120)
    if args.spatial:
        if args.no_bn:parser.error('Cambiar una variable: no combinar --spatial con --no-bn')
        cfg.update(CONVS_PER_BLOCK=2,RUN_NAME='F5_DIAG_MEMORIZE_SPATIAL')
    set_seed(cfg['SEED']);select_device(cfg)
    dataset = BreastDCEDataset(cfg['TRAIN_CSV'], cfg['DATA_DIR'], config=cfg)
    full = dataset.df.copy()
    dataset.df = select_patients(full, 2, cfg['SEED'])
    selected_ids = sorted(dataset.df.patient_id.unique().tolist())
    assert dataset.df.groupby('patient_id').size().to_dict() == full[full.patient_id.isin(selected_ids)].groupby('patient_id').size().to_dict()
    # Mismo orden aleatorio fijo en las comparaciones train/eval de BatchNorm.
    dataset.df = dataset.df.sample(frac=1, random_state=cfg['SEED']).reset_index(drop=True)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    images, labels = batch['image'], batch['label']
    assert torch.isfinite(images).all()
    rows = [dict(patient_id=str(p), sample_id=str(s), label=int(y))
            for p,s,y in zip(batch['patient_id'],batch['sample_id'],labels)]
    directory = create_run_directory(cfg);details=directory/'details';checkpoints=directory/'checkpoints'
    dataset.df.to_csv(details/'train_selection.csv', index=False)
    (details/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (details/'environment.json').write_text(json.dumps(environment_info(torch.device('cpu')),indent=2),encoding='utf-8')
    for source in ['src/model.py','src/dataset.py','src/configuration.py','src/phase5_diagnostics.py']:
        target=details/'sources'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    model = BreastCancerCNN(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['LEARNING_RATE'],weight_decay=cfg['WEIGHT_DECAY'])
    criterion=nn.BCEWithLogitsLoss()
    initial = probe(model,images,labels,rows)
    history=[]; gradients=[];started=time.time();passed=0
    generator=torch.Generator().manual_seed(cfg['SEED'])
    for epoch in range(1,121):
        model.train();total=0
        for ids in torch.randperm(len(images), generator=generator).split(cfg['BATCH_SIZE']):
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(images[ids]).squeeze(1),labels[ids]);loss.backward()
            norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
            if not torch.isfinite(loss) or not all(np.isfinite(list(norms.values()))):
                raise FloatingPointError('Loss o gradientes no finitos')
            if epoch in [1, 10, 40, 80, 120]:
                gradients.append(dict(epoch=epoch, **norms))
            optimizer.step();total+=float(loss.detach())*len(ids)
        if epoch==1 or epoch%5==0 or time.time()-started>=args.max_seconds:
            result=probe(model,images,labels,rows)
            history.append(dict(epoch=epoch,train_online_loss=total/len(images),eval_loss=result['loss'],
                patient_auc=result['patient']['auc'],slice_auc=result['slice']['auc'],
                patient_accuracy=result['patient']['accuracy'],slice_accuracy=result['slice']['accuracy'],
                seconds=time.time()-started))
            pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
            print(json.dumps(history[-1]),flush=True)
            passed=passed+1 if result['patient']['auc']>=.99 and result['slice']['accuracy']>=.90 and result['loss']<.25 else 0
            if passed>=2:break
        if time.time()-started>=args.max_seconds:break
    result=probe(model,images,labels,rows)
    batch_result=probe(model,images,labels,rows,batch_stats=True)
    calibrated=copy.deepcopy(model)
    calibrated.eval()
    for layer in calibrated.modules():
        if isinstance(layer,nn.BatchNorm2d):
            layer.reset_running_stats();layer.momentum=None;layer.train()
    with torch.no_grad():
        for chunk in images.split(cfg['BATCH_SIZE']):calibrated(chunk)
    calibrated_result=probe(calibrated,images,labels,rows)
    torch.save(dict(config=cfg,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,diagnostic_only=True,resume_supported=False),checkpoints/'last_epoch.pt')
    save_evaluation_artifacts(result,directory,details/'predictions')
    pd.DataFrame(gradients).to_csv(details/'gradient_norms.csv',index=False)
    info=dict(status='diagnostico_memorizacion_solo_train',run=directory.name,patient_ids=selected_ids,
              patients=len(selected_ids),slices=len(images),all_selected_patient_slices=True,seed=cfg['SEED'],
              initial_loss=initial['loss'],final_loss=result['loss'],epochs=epoch,seconds=time.time()-started,
              use_batch_norm=cfg['USE_BATCH_NORM'],convolutions_per_block=cfg['CONVS_PER_BLOCK'],dropout=cfg['DROPOUT'],weighted_loss=False,
              patient=result['patient'],slice=result['slice'],features=result['features'],spatial_features=result['spatial_features'],
              bn_batch_statistics=dict(loss=batch_result['loss'],patient=batch_result['patient'],slice=batch_result['slice']),
              bn_recalibrated_on_selected_train=dict(loss=calibrated_result['loss'],patient=calibrated_result['patient'],slice=calibrated_result['slice']),
              channel_ranges=[dict(min=float(x.min()),max=float(x.max()),std=float(x.std())) for x in images.unbind(1)],
              interpretation='Capacidad técnica de memorizar cuatro pacientes de train. No es AUC de validación ni se compara con runs completos.')
    (directory/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    fig,ax=plt.subplots();ax.plot([0]+[r['epoch'] for r in history],[initial['loss']]+[r['eval_loss'] for r in history],marker='o')
    ax.set(xlabel='Época',ylabel='BCE en pacientes memorizados',title='Diagnóstico: solo cuatro pacientes de train');fig.tight_layout();fig.savefig(directory/'loss_curve.png',dpi=150);plt.close(fig)
    path=session/('diagnostic_spatial.json' if args.spatial else ('diagnostic_no_bn.json' if args.no_bn else 'diagnostic_baseline.json'))
    path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
