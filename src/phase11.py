"""Fase 11 aislada: CNN4 con el entrenamiento y los folds oficiales existentes."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from src import phase10
from src.configuration import load_config, project_path, resolve_config
from src.cv_data import artifact_hash
from src.phase11_model import (ARCHITECTURE, BLOCK_CHANNELS, POOL_AFTER_BLOCK,
                               TRAINABLE_PARAMETERS, DCEFourBlockCNN)

CONFIG = "config/phase11/F11_CNN4.json"
SOURCES = ["src/phase11.py", "src/phase11_model.py", CONFIG]


def implementation():
    return {name: artifact_hash(project_path(name)) for name in SOURCES}


def template():
    cfg = resolve_config(load_config(CONFIG))
    cfg["PHASE11_SOURCES"] = implementation()
    verify_config(cfg)
    return cfg


def verify_config(cfg):
    expected = dict(ARCHITECTURE=ARCHITECTURE, BLOCK_CHANNELS=BLOCK_CHANNELS,
                    POOL_AFTER_BLOCK=POOL_AFTER_BLOCK, CONV_BIAS=False,
                    CONV_CHANNELS=[8, 16, 32, 64], INPUT_CHANNELS=3, DROPOUT=.30,
                    KERNEL_SIZE=3, PADDING=1, POOL_SIZE=2, CONVS_PER_BLOCK=2, NUM_CLASSES=1,
                    OPTIMIZER="adamw", LEARNING_RATE=.001, WEIGHT_DECAY=.0001,
                    SCHEDULER="plateau", SCHEDULER_FACTOR=.5, SCHEDULER_PATIENCE=3,
                    PRIMARY_METRIC="patient_auc", EVALUATE_TRAIN=True,
                    USE_WEIGHTED_LOSS=False, USE_AUGMENTATION=False,
                    INPUT_MODE="enhancement", USE_CLIPPING=True, USE_STANDARDIZATION=True,
                    USE_BATCH_NORM=True, SPATIAL_GRID=1, CLASSIFICATION_THRESHOLD=.5,
                    SEED=42, USE_EARLY_STOPPING=True, PATIENCE=8)
    differences = [key for key, value in expected.items() if cfg.get(key) != value]
    if differences:
        raise ValueError("Configuración F11 incompatible: " + ", ".join(differences))
    if not 1 <= cfg["EPOCHS"] <= 10:
        raise ValueError("F11 permite como máximo diez épocas")
    if not cfg["SMOKE_TEST"] and (cfg["IMAGE_SIZE"] != 256 or cfg["BATCH_SIZE"] != 16
            or cfg["MAX_TRAIN_BATCHES"] or cfg["MAX_VAL_BATCHES"]
            or cfg.get("TRAIN_PATIENTS_PER_CLASS")):
        raise ValueError("CV F11 requiere 256x256, batch 16 y particiones completas")
    if cfg.get("PHASE11_SOURCES") != implementation():
        raise ValueError("Las fuentes/configuración F11 cambiaron; no continuar esta campaña")


@contextmanager
def runtime(evaluation=False):
    """Enlazar la nueva factory solo en este proceso/contexto y restaurar al salir.

    Las fuentes protegidas no se editan. La delegación para otras configuraciones
    conserva exactamente el comportamiento y el contrato de sus CNN originales.
    """
    from src import checkpointing, experiment
    previous = dict(model=experiment.BreastCancerCNN,
                    contract=checkpointing.contract, snapshot=experiment.contract,
                    directory=experiment.create_run_directory, validate=experiment.validate)
    if evaluation:
        from src import evaluate
        previous.update(evaluator=evaluate.BreastCancerCNN, eval_validate=evaluate.validate)
    captured = {name: project_path(name).read_bytes() for name in SOURCES}
    created = []

    def model(cfg=None):
        return DCEFourBlockCNN(cfg) if (cfg or {}).get("ARCHITECTURE") == ARCHITECTURE else previous["model"](cfg)

    def contract(cfg):
        result = previous["contract"](cfg)
        if cfg.get("ARCHITECTURE") == ARCHITECTURE:
            verify_config(cfg)
            result["architecture"] = dict(id=ARCHITECTURE, blocks=BLOCK_CHANNELS,
                                          pool_after_block=POOL_AFTER_BLOCK, conv_bias=False)
            result["source_sha256"].update(implementation())
        return result

    def directory(cfg):
        target = previous["directory"](cfg)
        if cfg.get("ARCHITECTURE") == ARCHITECTURE:
            created.append(target)
        return target

    def validate(model, loader, criterion, device, config=None):
        result = previous["validate"](model, loader, criterion, device, config)
        if (config or {}).get("ARCHITECTURE") == ARCHITECTURE and not math.isfinite(result["patient"]["auc"]):
            raise ValueError("AUC paciente no válido: no avanzar scheduler ni seleccionar checkpoint F11")
        return result

    experiment.BreastCancerCNN = model
    if evaluation:
        evaluate.BreastCancerCNN = model
        evaluate.validate = validate
    checkpointing.contract = experiment.contract = contract
    experiment.create_run_directory = directory
    experiment.validate = validate
    try:
        yield
    finally:
        experiment.BreastCancerCNN = previous["model"]
        if evaluation:
            evaluate.BreastCancerCNN = previous["evaluator"]
            evaluate.validate = previous["eval_validate"]
        checkpointing.contract = previous["contract"]
        experiment.contract = previous["snapshot"]
        experiment.create_run_directory = previous["directory"]
        experiment.validate = previous["validate"]
        for target in created:
            # Completar el archivo técnico existente, también en intentos fallidos.
            with ZipFile(target/"internals/sources.zip", "a", compression=ZIP_DEFLATED) as archive:
                for name, content in captured.items():
                    archive.writestr(name, content)
            summary_path = target/"internals/summary.json"
            if summary_path.exists():
                summary = phase10.read(summary_path)
                summary.update(architecture=ARCHITECTURE, block_channels=BLOCK_CHANNELS,
                               convolutions_per_block=[1, 2, 2, 3],
                               pool_after_block=POOL_AFTER_BLOCK,
                               trainable_parameters=TRAINABLE_PARAMETERS)
                phase10.save(summary_path, summary)


def campaign_for(state_dir="models/phase11", campaign=None, create=False,
                 protocol=None, config=None):
    """Un registro persistente; repetir run no crea otra campaña."""
    from src.cross_validation import prepare_campaign, checked_campaign
    state_dir = project_path(state_dir)
    registry = state_dir/"route.json"
    if campaign:
        return checked(campaign)[0]
    if registry.exists():
        return checked(phase10.read(registry)["campaign"])[0]
    if not create:
        raise FileNotFoundError("No hay campaña F11 registrada; ejecutar primero run --folds 0")
    cfg = config or template()
    verify_config(cfg)
    parent = state_dir/"campaigns"
    candidates = sorted(parent.glob("CV9_*/campaign.json"))
    if len(candidates) > 1:
        raise ValueError("Varias campañas F11; revisar rutas sin crear otra")
    root = candidates[0].parent if candidates else prepare_campaign(
        "SPATIAL_SMALL", output_root=parent, protocol=protocol, template=cfg, smoke=cfg["SMOKE_TEST"])
    root, state, _ = checked_campaign(root)
    if phase10.read(root/"template.json") != cfg:
        raise ValueError("La campaña preparada no corresponde a esta configuración F11")
    if state["model"] == "SPATIAL_SMALL" and all(e["status"] == "prepared" for e in state["folds"].values()):
        # Solo la campaña NUEVA antes de ajustar/entrenar ningún fold.
        state.update(model=ARCHITECTURE, phase11_implementation=implementation())
        phase10.save(root/"campaign.json", state)
    checked(root)
    with ZipFile(root/"sources.zip", "a", compression=ZIP_DEFLATED) as archive:
        present = set(archive.namelist())
        for name in SOURCES:
            if name not in present:
                archive.write(project_path(name), name)
    phase10.save(registry, {"campaign": str(root.resolve())})
    return root


def checked(root):
    from src.cross_validation import checked_campaign
    root, state, frame = checked_campaign(root)
    if state["model"] != ARCHITECTURE or state.get("phase11_implementation") != implementation():
        raise ValueError("Campaña ajena o fuentes F11 incompatibles; no modificar campañas anteriores")
    verify_config(phase10.read(root/"template.json"))
    return root, state, frame


def train_job(cfg, resume=None):
    from src.experiment import train
    from src.check_suite import require_rocm
    from src.resume_validation import validate_resume_environment
    cfg = resolve_config(cfg)
    verify_config(cfg)
    if not cfg["SMOKE_TEST"]:
        if cfg["DEVICE"] != "cuda":
            raise RuntimeError("Entrenamiento clínico F11 requiere GPU ROCm; solo pruebas sintéticas en CPU")
        require_rocm()
    if resume:
        validate_resume_environment(project_path(resume))
    with runtime():
        target, _ = train(cfg, resume_from=resume)
    return target, phase10.read(target/"internals/summary.json")


def evaluate_job(checkpoint, output=None):
    from src.evaluate import evaluate_checkpoint
    payload = torch.load(project_path(checkpoint), map_location="cpu", weights_only=True)
    verify_config(payload["config"])
    with runtime(evaluation=True):
        return evaluate_checkpoint(checkpoint, output)


class Executor(phase10.Executor):
    """Usar el worker F11; conservar el aislamiento GPU y señales de Fase 10."""

    def __call__(self, args):
        args = list(args)
        if "src.train" in args:
            index = args.index("src.train")
            args[index:index+1] = ["src.phase11", "train"]
        return super().__call__(args)


def check_network(device):
    from src.experiment import set_seed, select_device
    from src.check_suite import require_rocm
    if device == "cuda":
        require_rocm()
    cfg = template()
    cfg["DEVICE"] = device
    actual = select_device(cfg)
    set_seed(cfg["SEED"])
    model = DCEFourBlockCNN(cfg).to(actual)
    logits = model(torch.randn(2, 3, 256, 256, device=actual))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([[0.], [1.]], device=actual))
    loss.backward()
    if (logits.shape != (2, 1) or not torch.isfinite(loss)
            or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())):
        raise RuntimeError("Forward/backward F11 inválido")
    if device == "cuda":
        torch.cuda.synchronize()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != TRAINABLE_PARAMETERS:
        raise ValueError("Recuento de parámetros diferente al protocolo F11")
    return dict(architecture=ARCHITECTURE, trainable_parameters=count, output_shape=list(logits.shape),
                device=str(actual), torch=str(torch.__version__), hip=torch.version.hip,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                forward_backward_passed=True, clinical_training_performed=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check", help="Forward/backward sintético sin entrenamiento clínico")
    check.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    for action in ["run", "status", "summarize"]:
        command = commands.add_parser(action)
        command.add_argument("--state-dir", default="models/phase11")
        command.add_argument("--campaign")
        if action == "run":
            command.add_argument("--folds", type=int, nargs="+", choices=range(5))
            command.add_argument("--restart-fold", type=int, nargs="+", choices=range(5), default=[])
    worker = commands.add_parser("train", help="Worker interno; usar run para gestionar los folds")
    source = worker.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume")
    worker.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    worker.add_argument("--epochs", type=int)
    evaluate = commands.add_parser("evaluate", help="Reevaluar el mejor checkpoint en su validación guardada")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--output", help="Carpeta nueva para resultados; no se sobrescribe")
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.action == "check":
        print(json.dumps(check_network(args.device), indent=2))
        return
    if args.action == "train":
        cfg = (torch.load(project_path(args.resume), map_location="cpu", weights_only=True)["config"]
               if args.resume else load_config(args.config))
        if args.epochs is not None and args.epochs != cfg["EPOCHS"]:
            arguments.error("No cambiar el objetivo de épocas al recuperar esta configuración F11")
        cfg = dict(cfg, DEVICE=args.device, DEADLINE_UNIX=None)
        train_job(cfg, args.resume)
        return
    if args.action == "evaluate":
        result = evaluate_job(args.checkpoint, args.output)
        print(json.dumps({k: result[k] for k in ["loss", "patient", "slice", "diagnostic_partial"]}, indent=2))
        return
    if args.action != "run":
        root = campaign_for(args.state_dir, args.campaign)
        with runtime():
            _, state, rows = phase10.inspect_campaign(root)
            print(json.dumps(rows, indent=2))
            if args.action == "summarize":
                from src.cross_validation import summarize_campaign
                if any(state["folds"][str(row["fold"])]["status"] != "complete" and row["status"] == "COMPLETO"
                       for row in rows):
                    raise ValueError("Manifiesto pendiente de reconciliar; repetir run (omite lo completo)")
                print(json.dumps(summarize_campaign(root), indent=2))
        return
    if not sys.platform.startswith("linux"):
        arguments.error("CV clínica F11 se ejecuta en Linux/ROCm; usar check --device cpu en Windows")
    logdir = project_path(args.state_dir)/"logs"
    logdir.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with phase10.gpu_lock(), (logdir/"F11_CNN4.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = phase10.Tee(console, log)
        try:
            executor = Executor()
            signal.signal(signal.SIGINT, executor.interrupt)
            signal.signal(signal.SIGTERM, executor.interrupt)
            if executor([sys.executable, "-u", "-m", "src.phase11", "check", "--device", "cuda"]):
                raise RuntimeError("Preflight F11/ROCm falló; no se inicia ningún fold")
            root = campaign_for(args.state_dir, args.campaign, create=True)
            _, state, _ = checked(root)
            if state["diagnostic"]:
                raise ValueError("No entrenar/interpretar una campaña diagnóstica como CV final")
            print("Campaña F11:", root, flush=True)
            with runtime():
                summary = phase10.run_manual(root, args.folds, args.restart_fold, executor)
            print(json.dumps(summary, indent=2))
        except (Exception, KeyboardInterrupt) as error:
            print("F11 DETENIDO / INCOMPLETO:", str(error), flush=True)
            raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1)
        finally:
            sys.stdout = console


if __name__ == "__main__":
    main()
