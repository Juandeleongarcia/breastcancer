"""Visualización agregada de umbrales OOF; no requiere datos individuales en hosting."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_components import charts
from app_components.layout import cards, plot
from app_services.thresholds import load_analysis


def analysis_for(catalog):
    try:
        return load_analysis(model_checksum=catalog['modalities']['multimodal']['checksum'])
    except (OSError,ValueError,KeyError,TypeError) as error:
        st.warning('No se puede utilizar la recomendación OOF: '+str(error))
        return None


def render(catalog):
    st.subheader('Umbral pCR · desarrollo OOF multimodal')
    report=analysis_for(catalog)
    if report is None:
        st.info('Análisis OOF no disponible. El umbral original sigue siendo 0,50.');return
    st.warning(report['methodology']['warning'])
    st.caption('Solo 1.097 pacientes de desarrollo, cada uno predicho por su fold excluido de train. '
               'El test externo de 176 pacientes permanece congelado a 0,50.')
    recommended=report['recommended_threshold']
    cards([('Pacientes OOF',report['n'],'Una predicción por paciente'),
           ('Umbral recomendado',f'{recommended:.2f}','Exploratorio; máximo Youden en la rejilla'),
           ('AUC OOF',f"{report['original']['auc']:.4f}",'No cambia al mover el umbral')])
    threshold=st.slider('Umbral para explorar OOF',.05,.95,float(recommended),.01,key='oof_threshold')
    selected=next(row for row in report['thresholds'] if abs(row['threshold']-threshold)<1e-10)
    rows=[]
    for key,label in [('threshold','Umbral'),('sensitivity','Sensibilidad'),('specificity','Especificidad'),
                      ('precision','Precisión pCR'),('f1','F1'),('balanced_accuracy','Balanced accuracy'),
                      ('accuracy','Accuracy'),('youden','Youden')]:
        fmt=(lambda v:f'{v:.4f}') if key in ['threshold','youden'] else (lambda v:f'{v:.2%}')
        rows.append(dict(Métrica=label,Original=fmt(report['original'][key]),
                         Recomendado=fmt(report['recommended'][key]),Explorado=fmt(selected[key])))
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    for col,label,row in zip(st.columns(2),['Original 0,50',f'Recomendado {recommended:.2f}'],
                             [report['original'],report['recommended']]):
        with col:
            st.caption(label+' · pacientes OOF')
            plot(charts.confusion_figure(row),'oof_cm_'+label)
    if threshold not in [.5,recommended]:
        with st.expander(f'Matriz del umbral explorado {threshold:.2f}',expanded=True):
            plot(charts.confusion_figure(selected),'oof_cm_selected')
    st.caption(charts.confusion_legend())
    plot(charts.threshold_curves(report,threshold),'oof_threshold_curves')
    plot(charts.threshold_curves(report,threshold,youden=True),'oof_youden_curve')
    st.caption('Rejilla fija 0,05–0,95, paso 0,01 (91 umbrales). Youden = sensibilidad + especificidad − 1. '
               'Empates: más próximo a 0,50 y después el mayor. Positivo si probabilidad ≥ umbral. '
               'La clasificación de la demostración se configura aparte; este control no la modifica.')
    with st.expander('Metodología y procedencia del análisis'):
        st.json({key:report[key] for key in ['methodology','source','model_checksum']})
    st.download_button('Descargar análisis agregado de umbrales OOF',
                       pd.DataFrame(report['thresholds']).drop(columns='confusion_matrix').to_csv(index=False).encode(),
                       file_name='umbrales_oof_desarrollo.csv',mime='text/csv')
