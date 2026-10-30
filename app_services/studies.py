"""Tripletas PNG en memoria. El orden z no aporta orientación o espaciado físico."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import PurePosixPath
import re
import stat
from zipfile import ZipFile, BadZipFile, ZIP_STORED, ZIP_DEFLATED

import numpy as np
from PIL import Image, UnidentifiedImageError

PHASES = ("PRE", "EARLY", "LATE")
MAX_BYTES = 128 * 1024 * 1024
MAX_SLICES = 256
PATTERN = re.compile(r"(?P<patient>.+)_z(?P<index>\d+)_(?P<phase>PRE|EARLY|LATE)\.png", re.I)


@dataclass(frozen=True)
class Study:
    patient_id: str
    slice_indices: tuple
    phases: np.ndarray  # [cortes, PRE/EARLY/LATE, alto, ancho], uint8, solo lectura
    synthetic: bool = False

    @property
    def fingerprint(self):
        digest = hashlib.sha256(self.phases.tobytes())
        digest.update(str((self.patient_id, self.slice_indices, self.synthetic)).encode())
        return digest.hexdigest()


def _safe_name(name):
    value = name.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("La ruta del archivo no es válida")
    return path.name


def study_from_files(files):
    """files: secuencia de (nombre, bytes), de un único paciente; no escribe disco."""
    if not files or len(files) > 3 * MAX_SLICES:
        raise ValueError("Aportar tripletas PRE/EARLY/LATE, con un máximo de 256 cortes")
    if sum(len(data) for _, data in files) > MAX_BYTES:
        raise ValueError("El estudio supera el límite de 128 MiB")
    patients = set()
    slices = {}
    for name, content in files:
        match = PATTERN.fullmatch(_safe_name(name))
        if not match:
            raise ValueError("Nombre incompatible. Usar PACIENTE_zNNN_PRE.png, EARLY.png y LATE.png")
        patient, index, phase = match.group("patient"), int(match.group("index")), match.group("phase").upper()
        patients.add(patient)
        if len(patients) != 1:
            raise ValueError("Cargar solamente un paciente por estudio")
        entry = slices.setdefault(index, {})
        if phase in entry:
            raise ValueError(f"Fase duplicada: corte {index}, {phase}")
        if len(content) > 1024 * 1024:
            raise ValueError("Un PNG supera el límite de 1 MiB")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (256, 256):
                    raise ValueError("Se requieren PNG de 8 bits en gris (L), de 256 × 256; no RGB ni 16 bits")
                entry[phase] = np.array(image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("No se pudo leer un PNG válido") from error
    for index, entry in slices.items():
        missing = set(PHASES)-entry.keys()
        if missing:
            raise ValueError(f"Corte {index}: faltan fases {', '.join(sorted(missing))}")
    indices = tuple(sorted(slices))
    array = np.stack([np.stack([slices[z][phase] for phase in PHASES]) for z in indices])
    array.setflags(write=False)
    return Study(next(iter(patients)), indices, array)


def study_from_uploads(files):
    """Uno o varios PNG, o un ZIP. Nunca se extraen rutas al sistema de archivos."""
    if len(files) == 1 and files[0][0].lower().endswith(".zip"):
        content = files[0][1]
        if len(content) > MAX_BYTES:
            raise ValueError("El ZIP supera el límite de 128 MiB")
        try:
            with ZipFile(BytesIO(content)) as archive:
                entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                if len(entries) > 3*MAX_SLICES or sum(e.file_size for e in entries) > MAX_BYTES:
                    raise ValueError("El contenido del ZIP supera los límites del estudio")
                values = []
                for entry in entries:
                    _safe_name(entry.filename)
                    if entry.compress_type not in [ZIP_STORED,ZIP_DEFLATED]:
                        raise ValueError('Usar un ZIP estándar sin compresión o con compresión Deflate')
                    if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                        raise ValueError("No se admiten enlaces ni archivos cifrados en el ZIP")
                    if entry.file_size > 1024*1024:
                        raise ValueError("Un archivo del ZIP supera 1 MiB")
                    values.append((entry.filename, archive.read(entry)))
                return study_from_files(values)
        except (BadZipFile,RuntimeError,NotImplementedError) as error:
            raise ValueError("El archivo ZIP no es válido") from error
    if any(name.lower().endswith(".zip") for name, _ in files):
        raise ValueError("Cargar un único ZIP o una selección de PNG, sin mezclarlos")
    return study_from_files(files)


def synthetic_study():
    """Formas geométricas deterministas: solo demostración del visor, no anatomía."""
    y, x = np.mgrid[-1:1:256j, -1:1:256j]
    rows = []
    for z in range(12):
        radius = .70 - .012*abs(z-5)
        ellipse = np.exp(-3*((x/radius)**2+(y/.73)**2))
        rings = .08*(np.cos(28*np.sqrt(x*x+y*y)+z*.2)+1)*ellipse
        spot = np.exp(-((x-.18)**2+(y+.11)**2)/.012) * np.exp(-((z-6)/3)**2)
        pre = .06 + .48*ellipse + rings
        early = pre + .18*spot + .035*ellipse
        late = pre + .11*spot + .055*ellipse
        rows.append(np.stack([np.clip(v*255, 0, 255).astype(np.uint8) for v in [pre,early,late]]))
    array = np.stack(rows)
    array.setflags(write=False)
    return Study("DEMO_SINTETICA", tuple(range(12)), array, synthetic=True)


def display_image(array, brightness=0., contrast=1.):
    """Ajuste de pantalla sobre una copia; no transforma los datos del modelo."""
    image = np.asarray(array, dtype=np.float32)/255.
    return np.clip((image-.5)*float(contrast)+.5+float(brightness), 0, 1)
