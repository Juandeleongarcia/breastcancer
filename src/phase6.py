"""Fase 6: benchmark real, admisión conservadora y cinco comparaciones controladas."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.batch_train import resolve_job, completed_run
from src.configuration import project_path, load_config, file_hash
from src.experiment import (train, select_device, set_seed, make_datasets,
                            create_criterion, create_optimizer, run_epoch, environment_info)
from src.model import BreastCancerCNN
from src.run_layout import technical_file, summary_paths
from src.checkpointing import experiment_key


def fits_budget(remaining_seconds, estimated_seconds, reserve=60):
    return remaining_seconds >= estimated_seconds + reserve


def measure(cfg, deadline):
    """Pesos desechables; nunca convierte métricas parciales en resultados clínicos."""
    set_seed(cfg['SEED'])
    device = select_device(cfg)
    train_ds, val_ds = make_datasets(cfg)
    model = BreastCancerCNN(cfg).to(device)
    optimizer = create_optimizer(model, cfg)
    criterion = create_criterion(cfg, device)
    measurements = {}
    for name, dataset, opt in [('train', train_ds, optimizer), ('validation', val_ds, None)]:
        loader = DataLoader(dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False,
                            num_workers=cfg['NUM_WORKERS'], pin_memory=device.type == 'cuda')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = run_epoch(model, loader, criterion, device, optimizer=opt, max_batches=4,
                           threshold=cfg['CLASSIFICATION_THRESHOLD'], deadline=deadline)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        samples = result['slice']['n']
        measurements[name] = dict(seconds=elapsed, measured_slices=samples,
                                  full_slices=len(dataset), full_patients=dataset.df.patient_id.nunique(),
                                  estimated_full_seconds=elapsed * len(dataset) / samples)
    epoch = sum(m['estimated_full_seconds'] for m in measurements.values())
    measurements.update(estimated_epoch_seconds=epoch,
                        conservative_run_seconds=cfg['EPOCHS'] * (epoch * 1.30 + 15),
                        maximum_epochs=cfg['EPOCHS'],
                        note='4 batches por fase, E/S y métricas incluidas; margen 30% + 15s/época. '
                             'Se presupuestan las 10 épocas: no se supone early stopping. '
                             'Benchmark descartable; sin AUC clínico comparable ni checkpoints.')
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return measurements


def audit(configs):
    reference = resolve_job('config/phase6/reference.json', 'cpu')
    tr = pd.read_csv(project_path(reference['TRAIN_CSV']))
    va = pd.read_csv(project_path(reference['VAL_CSV']))
    assert not set(tr.patient_id) & set(va.patient_id)
    assert tr.split.eq('train').all() and va.split.eq('train').all()
    assert not tr.sample_id.duplicated().any() and not va.sample_id.duplicated().any()
    assert tr.groupby('patient_id').pCR.nunique().eq(1).all()
    assert va.groupby('patient_id').pCR.nunique().eq(1).all()
    assert tr.patient_id.nunique() == 877 and va.patient_id.nunique() == 220
    partitions = {}
    for name, frame, csv in [('train', tr, reference['TRAIN_CSV']), ('validation', va, reference['VAL_CSV'])]:
        labels = frame[['patient_id', 'pCR']].drop_duplicates()
        partitions[name] = dict(patients=len(labels), slices=len(frame), sha256=file_hash(csv),
                                patient_classes={str(k): int(v) for k, v in labels.pCR.value_counts().items()},
                                majority_accuracy=float(labels.pCR.value_counts(normalize=True).max()))
    expected = {'WEIGHT_DECAY': {'WEIGHT_DECAY'}, 'DROPOUT_050': {'DROPOUT'},
                'SCHEDULER_FAST': {'SCHEDULER_PATIENCE'}, 'REGIONAL_2X2': {'SPATIAL_GRID'},
                'GENTLE_AUG': {'USE_AUGMENTATION', 'AUGMENTATION_MODE'}}
    differences = {}
    for name, cfg in configs.items():
        assert cfg['EPOCHS'] == 10 and cfg['PATIENCE'] == 5 and cfg['PRIMARY_METRIC'] == 'patient_auc'
        assert not cfg['SMOKE_TEST'] and not cfg['MAX_TRAIN_BATCHES'] and not cfg['MAX_VAL_BATCHES']
        assert cfg['STATS_PROVENANCE'] == 'verified_train_fit'
        changed = {k for k in reference if k not in {'RUN_NAME', 'DEVICE'} and reference[k] != cfg[k]}
        assert changed == expected[name], (name, changed)
        differences[name] = {k: {'reference': reference[k], 'experiment': cfg[k]} for k in sorted(changed)}
    local = []
    for path in summary_paths(project_path('models')):
        info = load_config(path)
        if (info.get('status') == 'entrenamiento_completo' and not info.get('diagnostic_partial')
                and 'validation' in info):
            local.append(dict(run=info['run'], auc_patient=info['validation']['patient']['auc']))
    return dict(partitions=partitions, differences=differences, local_full_runs=local,
                historical_reference=load_config('config/phase6/plan.json')['historical_reference'],
                patient_probability='media de probabilidades de todos los cortes del paciente',
                train_metric_protocol='online; pesos cambiantes, dropout y augmentation según configuración',
                validation_metric_protocol='fin de época, pesos fijos, eval, validación interna completa')


def summarize_run(directory, entry):
    summary = load_config(technical_file(directory, 'summary.json'))
    history = pd.read_csv(directory/'history.csv')
    best = history.loc[history.epoch.eq(summary['best_epoch'])].iloc[0]
    patient = summary['validation']['patient']
    predictions = pd.read_csv(directory/'internals/predictions_patient.csv')
    return dict(id=entry['id'], status=summary['status'], run=str(directory),
                epochs_completed=summary['epochs_completed'], best_epoch=summary['best_epoch'],
                train_loss=float(best.train_loss), validation_loss=summary['validation']['loss'],
                auc_patient=patient['auc'], auc_slice=summary['validation']['slice']['auc'],
                accuracy=patient['accuracy'], precision=patient['precision'], recall=patient['recall'],
                f1=patient['f1'], specificity=patient['specificity'],
                difference_from_reported_06360=patient['auc']-.6360,
                predicted_positive_fraction=float(predictions.prediction.mean()),
                probability_mean=float(predictions.probability.mean()),
                probability_std=float(predictions.probability.std(ddof=0)),
                final_train_loss=float(history.iloc[-1].train_loss),
                final_validation_loss=float(history.iloc[-1].val_loss),
                possible_overfitting=bool(history.iloc[-1].train_loss < history.iloc[0].train_loss
                                          and history.iloc[-1].val_loss > history.iloc[0].val_loss))


def save_report(session, state):
    columns = ['id', 'status', 'epochs_completed', 'best_epoch', 'train_loss', 'validation_loss',
               'auc_patient', 'auc_slice', 'accuracy', 'precision', 'recall', 'f1', 'specificity',
               'difference_from_reported_06360', 'estimated_minutes', 'run']
    frame = pd.DataFrame(state['results']).reindex(columns=columns)
    frame.to_csv(session/'comparison.csv', index=False)
    (session/'execution.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    text = '# Fase 6 — resultados verificables\n\n'
    text += ('Referencia 0,6360: dato comunicado por el usuario; no se presupone un checkpoint local. '
             'No se usa test ni se ajusta el umbral (0,50). Los valores ausentes no son resultados de rendimiento.\n\n')
    text += '```text\n'+frame.to_string(index=False)+'\n```\n\n'
    text += ('Train loss es online; no equivale a evaluar train con el checkpoint fijo. '
             'AUC y métricas principales son por paciente; AUC por corte se presenta por separado. '
             'Un descenso de train loss junto con aumento de validation loss es compatible con sobreajuste, '
             'pero también exige revisar calibración y desplazamiento de distribuciones. '
             'Accuracy cercana a la clase mayoritaria no prueba discriminación.\n\n')
    completed = [row for row in state['results'] if row.get('auc_patient') is not None]
    if completed:
        best = max(completed, key=lambda row: row['auc_patient'])
        text += f"Mejor AUC ejecutado: {best['id']} = {best['auc_patient']:.6f}.\n\n"
    else:
        text += ('No se completó ningún entrenamiento de la batería: no se puede determinar qué cambio mejora '
                 'o empeora el AUC ni elegir un ganador nuevo. El benchmark no es un experimento de rendimiento.\n\n')
    text += ('Prioridad provisional para GPU: WEIGHT_DECAY, SCHEDULER_FAST y REGIONAL_2X2. '
             'Primero regularización y dinámica; después contrastar pérdida de información espacial. '
             'Son prioridades técnicas, no una clasificación de eficacia. Mantener los cinco contrastes separados.\n')
    (session/'INFORME.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-minutes', type=float, default=120)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--session', help='Continuar la misma campaña y presupuesto tras revisar el benchmark')
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 120:
        parser.error('El presupuesto debe estar en (0, 120] minutos')
    entries = load_config('config/phase6/suite.json')['experiments']
    configs = {entry['id']: resolve_job(entry['config'], args.device) for entry in entries}
    device = select_device(next(iter(configs.values())))
    configs = {name: dict(cfg, DEVICE=str(device)) for name, cfg in configs.items()}
    if args.session:
        session = project_path(args.session)
        state = load_config(session/'execution.json')
        if state['configuration_keys'] != {name: experiment_key(cfg) for name, cfg in configs.items()}:
            raise ValueError('La configuración o fuentes cambiaron: no reutilizar este benchmark')
        if state['device'] != str(device):
            raise ValueError('El dispositivo cambió; realizar un benchmark nuevo')
    else:
        tag = datetime.now(ZoneInfo('Europe/Madrid')).strftime('%Y%m%d_%H%M%S')
        session = project_path('models')/f'PHASE6_{tag}_{uuid4().hex[:12]}'
        session.mkdir(exist_ok=False)
        started = time.time()
        state = dict(start_unix=started, deadline_unix=started+args.max_minutes*60,
                     maximum_minutes=args.max_minutes, device=str(device),
                     configuration_keys={name: experiment_key(cfg) for name, cfg in configs.items()},
                     benchmark={}, results=[], environment=environment_info(device))
        (session/'audit.json').write_text(json.dumps(audit(configs), indent=2), encoding='utf-8')
        print('Campaña:', session, flush=True)
        for entry in entries:
            name = entry['id']
            if time.time() >= state['deadline_unix']-60:
                break
            try:
                measured = measure(configs[name], state['deadline_unix']-60)
                state['benchmark'][name] = measured
                print(name, 'minutos estimados conservadores:', measured['conservative_run_seconds']/60, flush=True)
            except Exception:
                with (session/'errors.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(dict(id=name, stage='benchmark', traceback=traceback.format_exc()))+'\n')
            finally:
                save_report(session, state)
        state['benchmark_finished_unix'] = time.time()
    previous = {row['id']: row for row in state['results']}
    state['results'] = []
    try:
        for entry in entries:
            name = entry['id']
            cfg = configs[name]
            if previous.get(name, {}).get('auc_patient') is not None:
                state['results'].append(previous[name])
                continue
            estimate = state['benchmark'].get(name, {}).get('conservative_run_seconds')
            row = dict(id=name, status='pendiente_benchmark' if args.benchmark_only else 'no_ejecutado_presupuesto',
                       epochs_completed=0, estimated_minutes=estimate/60 if estimate else None)
            if not args.benchmark_only:
                done = completed_run(cfg)
                if done is not None:
                    row = summarize_run(done, entry)
                    row['status'] = 'omitido_completo'
                elif estimate is not None and fits_budget(state['deadline_unix']-time.time(), estimate):
                    cfg['DEADLINE_UNIX'] = state['deadline_unix']-60
                    try:
                        directory, summary = train(cfg)
                        row = (summarize_run(directory, entry) if summary.get('best_epoch')
                               else dict(row, status=summary['status'], run=str(directory)))
                    except Exception:
                        row['status'] = 'error'
                        with (session/'errors.jsonl').open('a', encoding='utf-8') as log:
                            log.write(json.dumps(dict(id=name, stage='training', traceback=traceback.format_exc()))+'\n')
            state['results'].append(row)
            save_report(session, state)
    finally:
        state['elapsed_wall_seconds'] = time.time()-state['start_unix']
        save_report(session, state)
    print('Informe:', session/'INFORME.md', flush=True)


if __name__ == '__main__':
    main()
