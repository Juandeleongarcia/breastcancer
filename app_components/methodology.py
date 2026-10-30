"""Arquitecturas verificadas y método experimental; sin generar mapas tumorales ficticios."""
import torch
from torch import nn
import streamlit as st

from app_components.layout import section,cards,resource,plot,empty
from app_components import charts
from app_services.lab import LABELS


def render():
    section('DE LA IMAGEN AL LOGIT','Arquitectura y metodología','CNN 2D propias, entrenadas desde cero. Formas y parámetros contrastados con los pesos publicados.')
    catalog=st.session_state.catalog
    if catalog is None:empty('Arquitecturas no verificadas','Se necesita el paquete de pesos y configuraciones reales.');return
    kind=st.radio('Arquitectura',['multimodal','image'],format_func=lambda k:LABELS[k],horizontal=True,key='architecture_kind')
    entry=catalog['modalities'][kind]
    try:bundle=resource(st.session_state.bundle_root,kind,entry['checksum'])
    except (OSError,ValueError,RuntimeError) as error:st.error(str(error));return
    member=bundle.members[0];cfg=member['config'];model=member['model']
    clinical_dim=member['clinical'].dimension if kind=='multimodal' else 0
    count=sum(p.numel() for p in model.parameters())
    cards([('Parámetros por CNN',f'{count:,}','Contados en PyTorch'),('Imagen','3 × 256 × 256','PRE · EARLY−PRE · LATE−PRE'),
        ('Vector de imagen','64','AdaptiveAvgPool 1 × 1'),('Vector clínico',str(clinical_dim),'Edad y subtipo' if clinical_dim else 'No utilizado')])
    st.caption('Los cinco checkpoints de esta modalidad tienen la misma arquitectura; sus pesos, estadísticas y mejores épocas son independientes.')
    stages=[('Entrada','3 × 256 × 256','Tres fases temporales; no RGB')]
    x=torch.zeros(1,3,256,256,device=bundle.device);block=1
    with torch.inference_mode():
        for layer in model.features:
            x=layer(x)
            if isinstance(layer,nn.MaxPool2d):
                stages.append((f'Bloque {block}',' × '.join(map(str,x.shape[1:])),'Dos Conv2d 3×3 + BatchNorm + ReLU; MaxPool 2×2'));block+=1
    stages.append(('Agregación','64','AdaptiveAvgPool 1×1 + Flatten'))
    if clinical_dim:stages.append(('Fusión',str(64+clinical_dim),'Concatenación de imagen y clínica'))
    stages.append(('Salida','1 logit',f'Dropout {cfg["DROPOUT"]:.2f} + Linear; Sigmoid en inferencia'))
    st.html('<div class="b-pipeline">'+''.join(f'<div class="b-step"><b>{i+1:02d}</b><strong>{name}</strong><span>{shape}</span></div>' for i,(name,shape,text) in enumerate(stages))+'</div>')
    selected=st.selectbox('Explorar una etapa',range(len(stages)),format_func=lambda i:stages[i][0],key=f'architecture_stage_{kind}')
    st.info(stages[selected][2])
    st.dataframe([dict(etapa=name,salida=shape,operaciones=text) for name,shape,text in stages],hide_index=True,width='stretch')
    if clinical_dim:
        st.subheader('Rama clínica y concatenación')
        st.write('Edad: imputación con mediana de train, z-score e indicador de ausencia. Subtipo: one-hot aprendido en train, con categorías ausentes y desconocidas explícitas. No se utilizan pCR, IDs, split, fold ni variables posteriores al tratamiento como predictores.')
        st.caption('Columnas codificadas del fold 0: '+', '.join(member['clinical'].columns))
        st.write('El vector clínico se repite para todos los cortes de la misma paciente. No se añade una MLP ni transfer learning: se concatena con las 64 características de imagen antes del clasificador lineal.')
    a,b=st.columns(2,gap='large')
    with a:
        st.subheader('Qué aprende una CNN')
        st.write('Las convoluciones extraen patrones espaciales locales. BatchNorm utiliza sus estadísticas aprendidas durante evaluación; ReLU introduce no linealidad y MaxPool reúne contexto reduciendo resolución. AdaptiveAvgPool produce una representación fija.')
        st.write('La salida es un logit. Sigmoid lo convierte en una puntuación de pCR, que no demuestra por sí sola calibración clínica.')
        st.subheader('Preprocesamiento idéntico al entrenamiento')
        st.write('PNG8 / 255 → PRE, EARLY−PRE, LATE−PRE → clipping por canal → estandarización. Cada miembro utiliza sus propios valores ajustados únicamente con train; nunca estadísticas calculadas con la muestra nueva.')
    with b:
        st.subheader('Cinco folds, cinco modelos nuevos')
        st.write('Cada paciente y todos sus cortes permanecen juntos. En cada fold se entrena desde cero con los otros cuatro y se selecciona el primer máximo de ROC AUC de validación por paciente.')
        st.write('Primero se promedian las probabilidades de los cortes por paciente y modelo. Después se promedian las cinco probabilidades con pesos iguales. No se promedian logits ni se votan clases.')
        st.subheader('Sobreajuste y fuga de información')
        st.write('Si baja train eval loss y sube validation loss, hay una señal de sobreajuste que debe examinarse. Estadísticas, categorías clínicas y selección de época se ajustan sin test. La validación retrospectiva no demuestra utilidad clínica prospectiva.')
    with st.expander('Configuración y trazabilidad'):
        st.dataframe([dict(fold=m['fold'],mejor_epoca=m['epoch'],parametros=m['parameters'],sha256=m['original_sha256'])
                      for m in entry['members']],hide_index=True,width='stretch')
        st.json({key:cfg[key] for key in ['OPTIMIZER','LEARNING_RATE','WEIGHT_DECAY','BATCH_SIZE','DROPOUT',
            'USE_WEIGHTED_LOSS','EPOCHS','PATIENCE','SEED','SCHEDULER']})
        st.caption('Entrenamiento original independiente de Streamlit. No se reentrena ni se recalibra desde la aplicación.')
    with st.expander('Clipping y normalización guardados por fold'):
        st.dataframe([dict(fold=i,p01=b['config']['CHANNEL_P01'],p99=b['config']['CHANNEL_P99'],
            medias=b['config']['CHANNEL_MEANS'],desviaciones=b['config']['CHANNEL_STDS'])
            for i,b in enumerate(bundle.members)],hide_index=True,width='stretch')
