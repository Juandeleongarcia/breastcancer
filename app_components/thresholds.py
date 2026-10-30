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
    rows=[dict(comparacion=label,**{key:row[key] for key in
          ['threshold','sensitivity','specificity','precision','f1','balanced_accuracy','accuracy','youden']})
          for label,row in [('Original',report['original']),('Recomendado OOF',report['recommended']),('Explorado',selected)]]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    for col,label,row in zip(st.columns(3),['Original 0,50',f'Recomendado {recommended:.2f}',f'Explorado {threshold:.2f}'],
                             [report['original'],report['recommended'],selected]):
        with col:
            st.caption(label+' · pacientes OOF')
            plot(charts.confusion_figure(row),'oof_cm_'+label)
    fig=go.Figure()
    for metric,label in [('sensitivity','Sensibilidad'),('specificity','Especificidad'),
                         ('precision','Precisión pCR'),('f1','F1'),('balanced_accuracy','Balanced accuracy'),('youden','Youden')]:
        fig.add_scatter(x=[r['threshold'] for r in report['thresholds']],
                        y=[r[metric] for r in report['thresholds']],name=label,mode='lines')
    fig.add_vline(x=.5,line_dash='dot',annotation_text='Original 0,50')
    fig.add_vline(x=recommended,line_dash='dash',line_color=charts.TEAL,annotation_text=f'OOF {recommended:.2f}')
    fig.add_vline(x=threshold,line_color=charts.WHITE)
    charts.style(fig,450,'Métricas por umbral · desarrollo OOF')
    fig.update_xaxes(title='Umbral pCR',range=[.05,.95]);fig.update_yaxes(title='Valor de la métrica',range=[-1,1])
    plot(fig,'oof_threshold_curves')
    st.caption('Rejilla fija 0,05–0,95, paso 0,01 (91 umbrales). Youden = sensibilidad + especificidad − 1. '
               'Empates: más próximo a 0,50 y después el mayor. Positivo si probabilidad ≥ umbral. '
               'La clasificación de la demostración se configura aparte; este control no la modifica.')
    with st.expander('Metodología y procedencia del análisis'):
        st.json({key:report[key] for key in ['methodology','source','model_checksum']})
    st.download_button('Descargar análisis agregado de umbrales OOF',
                       pd.DataFrame(report['thresholds']).drop(columns='confusion_matrix').to_csv(index=False).encode(),
                       file_name='umbrales_oof_desarrollo.csv',mime='text/csv')
