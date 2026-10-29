"""Transformación clínica train-only, serializable en JSON, sin etiquetas predictoras."""
import hashlib
import json

import numpy as np
import pandas as pd

FEATURES = ('HR_HER2_STATUS', 'age')
MISSING = '__MISSING__'
UNKNOWN = '__UNKNOWN__'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selected_patients(patients, ids):
    ids = list(map(str, ids))
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Se requieren identificadores únicos no vacíos')
    required = ['pid', *FEATURES]
    if any(key not in patients for key in required) or patients.pid.isna().any() or patients.pid.duplicated().any():
        raise ValueError('Clínica incompleta o pacientes duplicados')
    lookup = patients.assign(pid=patients.pid.astype(str)).set_index('pid')
    if not set(ids).issubset(lookup.index):
        raise ValueError('Faltan pacientes clínicos; no se permite un join que pierda filas')
    return lookup.loc[ids, list(FEATURES)].copy()


def category(value):
    if pd.isna(value) or not str(value).strip():
        return MISSING
    value = str(value).strip()
    if value in [MISSING, UNKNOWN]:
        raise ValueError('Categoría reservada en el archivo de origen')
    return value


def ages(frame):
    result = pd.to_numeric(frame.age, errors='raise').to_numpy(dtype=float, na_value=np.nan)
    known = result[~np.isnan(result)]
    if not np.isfinite(known).all() or ((known < 0) | (known > 120)).any():
        raise ValueError('Edad no numérica, no finita o fuera de 0–120 años')
    return result


class ClinicalTransform:
    """Edad z-score + indicador ausente + one-hot con buckets explícitos."""
    def __init__(self, state):
        self.state = state
        categories = state['categories']
        if (state.get('schema_version') != 1 or state.get('features') != list(FEATURES)
                or categories != sorted(set(categories)) or any(c in [MISSING,UNKNOWN] for c in categories)
                or not np.isfinite([state['age_median'], state['age_mean'], state['age_std']]).all()
                or state['age_std'] <= 0):
            raise ValueError('Transformador clínico incompatible')
        self.columns = ['age_z', 'age_missing', *['subtype='+c for c in categories+[MISSING, UNKNOWN]]]
        if state['columns'] != self.columns:
            raise ValueError('Orden de codificación clínica alterado')

    @classmethod
    def fit(cls, patients, train_ids, validation_ids=(), external_ids=()):
        train_ids = list(map(str, train_ids))
        if set(train_ids) & (set(map(str, validation_ids)) | set(map(str, external_ids))):
            raise ValueError('Fuga de pacientes al ajustar clínica')
        frame = selected_patients(patients, train_ids)
        values = ages(frame)
        if np.isnan(values).all():
            raise ValueError('No hay edades observadas en entrenamiento')
        median = float(np.nanmedian(values))
        filled = np.where(np.isnan(values), median, values)
        cats = sorted(set(frame.HR_HER2_STATUS.map(category)) - {MISSING})
        state = dict(schema_version=1, features=list(FEATURES), categories=cats,
            age_median=median, age_mean=float(filled.mean()), age_std=float(filled.std()) or 1.,
            n_train_patients=len(train_ids), train_ids_sha256=digest(sorted(train_ids)),
            columns=['age_z','age_missing', *['subtype='+c for c in cats+[MISSING,UNKNOWN]]],
            fit_scope='unique_training_patients_only', tum_vol='excluded_unverified_baseline_provenance')
        return cls(state)

    @property
    def dimension(self):
        return len(self.columns)

    def transform(self, patients, ids):
        frame = selected_patients(patients, ids)
        numeric = ages(frame)
        missing = np.isnan(numeric)
        filled = np.where(missing, self.state['age_median'], numeric)
        result = np.zeros((len(frame), self.dimension), dtype=np.float32)
        result[:,0] = (filled-self.state['age_mean'])/self.state['age_std']
        result[:,1] = missing
        vocabulary = self.state['categories']+[MISSING, UNKNOWN]
        for i,value in enumerate(frame.HR_HER2_STATUS.map(category)):
            encoded = value if value in vocabulary else UNKNOWN
            result[i,2+vocabulary.index(encoded)] = 1.
        if not np.isfinite(result).all():
            raise ValueError('Codificación clínica no finita')
        return result


def feature_hash(patients, ids):
    frame = selected_patients(patients, sorted(map(str,ids)))
    values = ages(frame)
    return digest([[pid,category(row.HR_HER2_STATUS), None if np.isnan(age) else float(age)]
                   for (pid,row),age in zip(frame.iterrows(),values)])
