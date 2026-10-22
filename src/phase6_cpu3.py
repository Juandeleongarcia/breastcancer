"""Campaña CPU separada: hasta tres runs completos de tres épocas, sin editar Fase 6 GPU."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.batch_train import resolve_job, completed_run
from src.configuration import load_config, project_path
from src.experiment import train, select_device, environment_info
from src.phase6 import audit, measure, fits_budget, summarize_run, save_report


PRIORITY = ['REGIONAL_2X2', 'WEIGHT_DECAY', 'GENTLE_AUG']


def cpu_config(entry):
    cfg = resolve_job(entry['config'], 'cpu')
    cfg.update(EPOCHS=3, RUN_NAME='F6_CPU3_'+entry['id'],
               INTERPRETATION='preliminar_tres_epocas_train_y_validacion_completos')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-minutes', type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 120:
        parser.error('El presupuesto debe estar en (0, 120] minutos')
    entries = {entry['id']: entry for entry in load_config('config/phase6/suite.json')['experiments']}
    originals = {name: resolve_job(entry['config'], 'cpu') for name, entry in entries.items()}
    verified = audit(originals)
    configs = {name: cpu_config(entries[name]) for name in PRIORITY}
    device = select_device(configs[PRIORITY[0]])
    tag = datetime.now(ZoneInfo('Europe/Madrid')).strftime('%Y%m%d_%H%M%S')
    session = project_path('models')/f'PHASE6_CPU3_{tag}_{uuid4().hex[:12]}'
    session.mkdir(exist_ok=False)
    started = time.time()
    state = dict(start_unix=started, deadline_unix=started+args.max_minutes*60,
                 maximum_minutes=args.max_minutes, device=str(device),
                 benchmark={}, results=[], environment=environment_info(device),
                 priority=PRIORITY, runtime_configs=configs, dataset_scope='train_y_validacion_completos',
                 protocol_note='3 épocas CPU preliminares; configuraciones GPU originales intactas a 10 épocas')
    (session/'audit.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')

    def persist():
        state['elapsed_wall_seconds'] = time.time()-started
        save_report(session, state)
        with (session/'INFORME.md').open('a', encoding='utf-8') as report:
            report.write('\nCampaña CPU independiente de TRES épocas: resultados preliminares con TODO train '
                         'y validación interna, no subconjuntos. Las configuraciones originales siguen en diez '
                         'épocas. Train loss es online; cualquier comparación con 0,6360 es histórica, '
                         'sin atribución causal ni garantía de superioridad.\n')

    print('Campaña CPU3:', session, flush=True)
    for name in PRIORITY:
        try:
            state['benchmark'][name] = measure(configs[name], state['deadline_unix']-120)
            print(name, 'estimación conservadora 3 épocas (min):',
                  state['benchmark'][name]['conservative_run_seconds']/60, flush=True)
        except Exception:
            with (session/'errors.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(dict(id=name, stage='benchmark', traceback=traceback.format_exc()))+'\n')
        finally:
            persist()
    try:
        for name in PRIORITY:
            cfg = configs[name]
            estimate = state['benchmark'].get(name, {}).get('conservative_run_seconds')
            remaining = state['deadline_unix']-time.time()
            row = dict(id=name, status='no_ejecutado_presupuesto', epochs_completed=0,
                       estimated_minutes=estimate/60 if estimate else None)
            done = completed_run(cfg)
            if done:
                row = summarize_run(done, entries[name])
                row['status'] = 'omitido_completo'
            elif estimate is not None and fits_budget(remaining, estimate, reserve=120):
                cfg['DEADLINE_UNIX'] = state['deadline_unix']-120
                state['active_experiment'] = name
                persist()
                print('ADMITIDO:', name, 'restante (min):', remaining/60, flush=True)
                try:
                    directory, summary = train(cfg)
                    row = (summarize_run(directory, entries[name]) if summary.get('best_epoch')
                           else dict(row, status=summary['status'], run=str(directory)))
                    row['estimated_minutes'] = estimate/60
                    row['interpretation'] = f"preliminar_{row.get('epochs_completed', 0)}_de_3_epocas_dataset_completo"
                except Exception:
                    row['status'] = 'error'
                    with (session/'errors.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(dict(id=name, stage='training', traceback=traceback.format_exc()))+'\n')
                finally:
                    state.pop('active_experiment', None)
            else:
                print('PENDIENTE:', name, 'no cabe con margen; restante (min):', remaining/60, flush=True)
            state['results'].append(row)
            persist()
        for name in entries:
            if name not in PRIORITY:
                state['results'].append(dict(id=name, status='pendiente_GPU_no_priorizado_CPU', epochs_completed=0))
    finally:
        persist()
    print('Informe CPU3:', session/'INFORME.md', flush=True)


if __name__ == '__main__':
    main()
