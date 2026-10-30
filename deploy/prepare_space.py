"""Preparar solo código autorizado para HF Spaces; sin dataset, etiquetas o pesos."""
import argparse
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def prepare(destination,target='space'):
    if target not in ['space','streamlit']:raise ValueError('Destino de publicación desconocido')
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('La salida ya existe; elegir otra para no sobrescribirla')
    destination.mkdir(parents=True)
    files=['app.py','config/config.py','config/app_release.json','config/app_oof_threshold.json','config/app_identity.json','.streamlit/config.toml',
           'deploy/app.py','deploy/requirements.txt','ATTRIBUTION.md','LICENSE']
    for folder,pattern in [('app_components','*.py'),('app_services','*.py'),('src','*.py'),('assets','*.css'),('assets','*.svg')]:
        files.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob(pattern))
    for name in files:
        file_target=destination/name;file_target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,file_target)
    shutil.copyfile(ROOT/'deploy/Dockerfile',destination/'Dockerfile')
    (destination/'.gitignore').write_text('models/\n.venv/\nvenv/\nenv/\n.env\n.env.*\n__pycache__/\n*.pyc\n*.log\n.pytest_cache/\n.streamlit/secrets.toml\nmetadata/\ndataset/\nbreastdcedl/\nbackups/\ncheckpoints/\nweights/\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.onnx\n*.pkl\n*.joblib\n*.csv\n*.png\n*.jpg\n*.jpeg\n*.dcm\n*.dicom\n*.nii\n*.nii.gz\n*.mha\n*.nrrd\n*.zip\n',encoding='utf-8')
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
    if target=='streamlit':
        shutil.copyfile(ROOT/'deploy/README_APP.md',destination/'README.md')
        (destination/'tests').mkdir()
        shutil.copyfile(ROOT/'tests/test_app_delivery.py',destination/'tests/test_app_delivery.py')
    archive=destination.with_suffix('.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as stream:
        for path in sorted(destination.rglob('*')):
            if path.is_file():stream.write(path,path.relative_to(destination).as_posix())
    print('Carpeta de código limpio ('+target+'):',destination)
    print('ZIP de código autorizado:',archive)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--target',choices=['space','streamlit'],default='space')
    args=parser.parse_args();prepare(args.output,args.target)
