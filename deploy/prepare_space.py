"""Preparar solo código autorizado para HF Spaces; sin dataset, etiquetas o pesos."""
import argparse
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('La salida ya existe; elegir otra para no sobrescribirla')
    destination.mkdir(parents=True)
    files=['app.py','config/config.py','config/app_release.json','.streamlit/config.toml',
           'deploy/requirements.txt','ATTRIBUTION.md','LICENSE']
    for folder,pattern in [('app_components','*.py'),('app_services','*.py'),('src','*.py'),('assets','*.css'),('assets','*.svg')]:
        files.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob(pattern))
    for name in files:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(ROOT/'deploy/Dockerfile',destination/'Dockerfile')
    (destination/'README.md').write_text('''---
title: BreastDCEDL AI Lab
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
---

# BreastDCEDL AI Lab — predicción de respuesta pCR

Prototipo universitario de investigación. No destinado a diagnóstico o decisiones clínicas.

La aplicación descarga artefactos de inferencia propios con SHA256 fijo. No incluye
datasets, CSV con etiquetas por paciente ni validación privada. Las imágenes subidas
se procesan en memoria de la sesión y no se almacenan de forma persistente.

Datos del ejemplo público: BreastDCEDL MinCrop, Fridman et al., I-SPY1/I-SPY2/Duke.
CC BY-NC 4.0. DOI: https://doi.org/10.5281/zenodo.18114231
Ver ATTRIBUTION.md. Cinco muestras de la defensa no constituyen validación clínica.
''',encoding='utf-8')
    archive=destination.with_suffix('.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as stream:
        for path in sorted(destination.rglob('*')):
            if path.is_file():stream.write(path,path.relative_to(destination).as_posix())
    print('Carpeta HF Spaces:',destination)
    print('ZIP de código autorizado:',archive)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)
