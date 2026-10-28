"""Contrastes Fase 8 y comparación de resultados reales, sin alterar controles."""
import argparse
from datetime import datetime
import hashlib
import json
from uuid import uuid4

import pandas as pd

from src.configuration import load_config, project_path, resolve_config, csv_hash_variants
from src.checkpointing import COMPATIBILITY_KEYS
from src.run_layout import summary_paths, technical_file


CONTRASTS = {
    'F8_SMALL_LR': ('config/phase7/SPATIAL_SMALL.json', 'LEARNING_RATE', .0001),
    'F8_SMALL_DROPOUT': ('config/phase7/SPATIAL_SMALL.json', 'DROPOUT', .5),
    'F8_CONTROL_LR': ('config/phase7/SPATIAL_CONTROL.json', 'LEARNING_RATE', .0001),
}
STATISTICS_KEYS = ['INPUT_MODE', 'USE_CLIPPING', 'CHANNEL_P01', 'CHANNEL_P99',
                   'USE_STANDARDIZATION', 'CHANNEL_MEANS', 'CHANNEL_STDS']


def statistics_signature(cfg):
    return hashlib.sha256(json.dumps({k: cfg[k] for k in STATISTICS_KEYS}, sort_keys=True).encode()).hexdigest()


def audit_job(entry, cfg=None):
    """Bloquear solo el trabajo afectado por diferencias de control/config/stats.

    La ausencia de runs de clase se informa, sin fingir una réplica histórica.
    Si están presentes se contrastan sus snapshots, no se sustituyen sus stats.
    """
    from src.batch_train import resolve_job
    name = entry['id']
    if name not in CONTRASTS:
        raise ValueError('Experimento Fase 8 desconocido: '+name)
    reference_path, variable, value = CONTRASTS[name]
    if entry.get('control_config') != reference_path or entry.get('expected_changes') != {variable: value}:
        raise ValueError('Referencia/contraste distinto del acordado: '+name)
    cfg = cfg if cfg is not None else resolve_job(entry['config'], 'cpu')
    control = resolve_job(reference_path, cfg['DEVICE'])
    excluded = {'RUN_NAME', 'COMPARISON_PARENT'}
    differences = {k for k in set(cfg) | set(control) if k not in excluded and cfg.get(k) != control.get(k)}
    # DEADLINE_UNIX es un límite externo de ejecución, no un hiperparámetro.
    differences.discard('DEADLINE_UNIX')
    if differences != {variable} or cfg[variable] != value:
        raise ValueError(f'Diferencias no autorizadas respecto a {reference_path}: {sorted(differences)}')
    if cfg['RUN_NAME'] != name or cfg['COMPARISON_PARENT'] != control['RUN_NAME']:
        raise ValueError('Identidad o referencia incorrecta: '+name)
    if (cfg['EPOCHS'] != 20 or cfg['PATIENCE'] != 8 or not cfg['USE_EARLY_STOPPING']
            or not cfg['EVALUATE_TRAIN'] or cfg['PRIMARY_METRIC'] != 'patient_auc'
            or cfg['SMOKE_TEST'] or cfg.get('TRAIN_PATIENTS_PER_CLASS')
            or cfg['MAX_TRAIN_BATCHES'] or cfg['MAX_VAL_BATCHES']
            or max(cfg['CONV_CHANNELS']) > 128 or cfg['INPUT_CHANNELS'] != 3
            or cfg['CLASSIFICATION_THRESHOLD'] != .5):
        raise ValueError('Protocolo Fase 8 incompatible: '+name)
    signature = statistics_signature(control)
    if signature != entry.get('reference_statistics_sha256') or statistics_signature(cfg) != signature:
        raise ValueError('Estadísticas distintas del control firmado; no se sustituyen: '+name)
    cfg_paths = []
    if entry.get('control_run'):
        cfg_paths = [technical_file(project_path(entry['control_run']), 'config.json')]
        if not cfg_paths[0].is_file():
            raise FileNotFoundError('No existe config.json del control explícito')
    else:
        for path in summary_paths(project_path(cfg['RESULTS_DIR'])):
            summary = load_config(path)
            if summary.get('experiment') == control['RUN_NAME'] and not summary.get('diagnostic_partial', True):
                directory = path.parent.parent if path.parent.name == 'internals' else path.parent
                cfg_paths.append(technical_file(directory, 'config.json'))
    compared = []
    for path in cfg_paths:
        saved = resolve_config(load_config(path))
        keys = set(COMPATIBILITY_KEYS) | {'EVALUATE_TRAIN', 'TRAIN_PATIENTS_PER_CLASS', 'EPOCHS'}
        changed = sorted(k for k in keys if saved.get(k) != control.get(k))
        if saved['RUN_NAME'] != control['RUN_NAME'] or changed:
            raise ValueError(f'Control histórico incompatible {path}: {changed}; detener solo {name}')
        # CSV de clase puede tener CRLF; no se adivinan particiones distintas.
        for key in ['TRAIN_CSV', 'VAL_CSV']:
            if not (csv_hash_variants(saved[key]) & csv_hash_variants(control[key])):
                raise ValueError('Partición histórica distinta: '+str(path))
        compared.append(str(path))
    return dict(id=name, control=reference_path, variable=variable,
                before=control[variable], after=value,
                reference_statistics_sha256=signature, historical_controls=compared,
                historical_comparability='snapshot_config_verificado' if compared else 'control_de_clase_no_disponible_no_reproducido')


def verify_phase8_suite(path, device='cpu', synthetic=True, only=None):
    from src.batch_train import resolve_job
    from src.check_suite import verify_partitions, check_network
    suite = load_config(path)
    entries = suite['experiments']
    ids = [e['id'] for e in entries]
    if len(ids) != 3 or set(ids) != set(CONTRASTS):
        raise ValueError('Se requieren exactamente los tres experimentos Fase 8')
    if only and set(only)-set(ids):
        raise ValueError('Identificador --only desconocido')
    checks, errors = [], []
    for entry in entries:
        if only and entry['id'] not in only:
            continue
        try:
            cfg = resolve_job(entry['config'], device)
            audited = audit_job(entry, cfg)
            partitions = verify_partitions(cfg)
            parameters = check_network(cfg) if synthetic else None
            checks.append(dict(**audited, parameters=parameters, partitions=partitions, passed=True))
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
            errors.append(dict(id=entry['id'], error=str(error)))
    return dict(models=checks, errors=errors, passed=not errors, synthetic_only=True,
                clinical_training_performed=False)


def phase8_comparison(root='models'):
    from src.compare_runs import comparison
    names = list(CONTRASTS) + ['F7_SPATIAL_SMALL', 'F7_SPATIAL_CONTROL']
    frame = comparison(root)
    if 'Experimento' not in frame:
        frame['Experimento'] = None
    selected = frame[frame.Experimento.isin(names)].copy()
    if 'Grupo_comparable' in selected:
        selected = selected[selected.Grupo_comparable.eq('train_completo')]
    from src.batch_train import resolve_job
    for index, row in selected.iterrows():
        try:
            cfg_path = (f'config/phase8/{row.Experimento}.json' if row.Experimento in CONTRASTS else
                        f'config/phase7/{row.Experimento.removeprefix("F7_")}.json')
            expected = resolve_job(cfg_path, 'cpu')
            saved = resolve_config(load_config(technical_file(project_path(root)/row.Run, 'config.json')))
            keys = set(COMPATIBILITY_KEYS) | {'EVALUATE_TRAIN','EPOCHS','TRAIN_PATIENTS_PER_CLASS'}
            changed = sorted(k for k in keys if saved.get(k) != expected.get(k))
            for key in ['TRAIN_CSV','VAL_CSV']:
                if not (csv_hash_variants(saved[key]) & csv_hash_variants(expected[key])):
                    changed.append(key)
            selected.loc[index,'Comparabilidad'] = 'config_difiere_revisar' if changed else 'config_coincide_referencia_versionada'
            selected.loc[index,'Diferencias_config'] = ', '.join(changed)
        except (ValueError, FileNotFoundError, KeyError) as error:
            selected.loc[index,'Comparabilidad'] = 'config_no_verificable'
            selected.loc[index,'Diferencias_config'] = str(error)
    placeholders = [dict(Experimento=name, Estado='pendiente_GPU' if name in CONTRASTS else 'control_no_disponible_localmente')
                    for name in names if name not in set(selected.Experimento)]
    result = pd.concat([selected, pd.DataFrame(placeholders)], ignore_index=True)
    required = ['Experimento','Estado','Comparabilidad','Diferencias_config','Run','Epocas','Mejor_epoca','ROC_AUC_paciente','ROC_AUC_corte',
                'Val_loss','Train_eval_loss_mejor_epoca','Loss_gap_mejor_epoca','Accuracy_paciente',
                'Precision_paciente','Sensibilidad_paciente','Especificidad_paciente','F1_paciente',
                'Balanced_accuracy_paciente','PR_AUC_AP_paciente','Parametros_entrenables',
                'Tiempo_total_s','Evidencia_sobreajuste']
    return result.reindex(columns=required+[k for k in result if k not in required])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='models')
    parser.add_argument('--output')
    args = parser.parse_args()
    frame = phase8_comparison(args.root)
    output = project_path(args.output) if args.output else project_path(args.root)/(
        f'PHASE8_COMPARISON_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:12]}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)
    print(frame.to_string(index=False))
    print('Comparación Fase 8:', output)


if __name__ == '__main__':
    main()
