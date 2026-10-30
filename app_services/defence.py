"""Cinco pruebas privadas en memoria. El CSV no contiene imágenes, IDs ni clínica."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from uuid import uuid4

import pandas as pd


class Defence:
    def __init__(self):
        self.slot=1;self.records=[];self.seen=set();self.current=None
        self.session_id=uuid4().hex

    def input_key(self,study,kind,clinical):
        return hashlib.sha256((study.fingerprint+kind+json.dumps(clinical,sort_keys=True)).encode()).hexdigest()

    def record(self,result,study,key):
        if self.current is not None:raise ValueError('La muestra ya está registrada; pasar a la siguiente')
        if len(self.records)>=5:raise ValueError('Las cinco muestras ya están completadas')
        if study.fingerprint in self.seen:raise ValueError('Esta misma imagen ya se analizó en otra muestra de la sesión')
        row=dict(muestra=self.slot,timestamp=datetime.now(ZoneInfo('Europe/Madrid')).isoformat(timespec='seconds'),
            modelo=result['model'],checksum=result['checksum'],umbral=result['threshold'],
            probabilidad_pcr=result['patient_probability'],clase_predicha=result['prediction'],
            modelos=result['n_models'],cortes=result['n_slices'],dispositivo=result['device'],
            segundos=result['seconds'],etiqueta_real=None,
            origen_umbral=result.get('threshold_policy','Original 0,50'),
            sesion_id=self.session_id,ejecucion_id=uuid4().hex)
        self.records.append(row);self.seen.add(study.fingerprint);self.current=key
        return row

    def reveal(self,label):
        if self.current is None or label not in [0,1]:raise ValueError('Revelar etiqueta solo después de predecir')
        old=self.records[-1]['etiqueta_real']
        if old is not None and old!=label:raise ValueError('La etiqueta revelada ya está registrada')
        self.records[-1]['etiqueta_real']=int(label)

    def next(self):
        if self.current is None:raise ValueError('Analizar la muestra actual antes de continuar')
        if self.slot>=5:raise ValueError('Se han completado las cinco muestras')
        self.slot+=1;self.current=None

    def csv(self):return pd.DataFrame(self.records).to_csv(index=False).encode('utf-8-sig')
