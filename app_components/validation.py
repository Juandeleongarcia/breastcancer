"""Solo resultados agregados verificados; no existe un botón para reevaluar test."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_components.layout import section,cards,plot,empty
from app_components import charts
from app_services.results import aggregate_histories
from app_services.lab import safe_file,LABELS


def roc_plot(entry,external=False):
    fig=go.Figure()
    if external:
        series=[('Test externo congelado',entry['roc'],entry['metrics']['auc'],charts.TEAL)]
    else:
        series=[(f'Fold {f["fold"]}',f['roc'],f['metrics']['auc'],color) for f,color in zip(entry['folds'],
            [charts.TEAL,charts.PURPLE,'#85ABD4','#D9B878','#C999AF'])]
        series.append(('OOF interno',entry['oof']['roc'],entry['oof']['metrics']['auc'],charts.WHITE))
    for name,data,auc,color in series:
        fig.add_scatter(x=data['fpr'],y=data['tpr'],mode='lines',name=f'{name} · AUC {auc:.4f}',line=dict(color=color))
    fig.add_scatter(x=[0,1],y=[0,1],mode='lines',name='Referencia aleatoria',line=dict(color=charts.MUTED,dash='dot'))
    charts.style(fig,440);fig.update_layout(hovermode='closest',legend=dict(orientation='h',y=-.25))
    fig.update_xaxes(title='1 − especificidad',range=[0,1]);fig.update_yaxes(title='Sensibilidad',range=[0,1])
    return fig


def metric_cards(metrics):
    cards([(label,f'{metrics[key]:.1%}',detail) for label,key,detail in [
        ('Accuracy','accuracy','Clasificación global'),('Balanced accuracy','balanced_accuracy','Media de sensibilidad y especificidad'),
        ('Sensibilidad pCR','recall','TP / (TP + FN)'),('Especificidad','specificity','TN / (TN + FP)')]])


def render():
    section('EVIDENCIA EXPERIMENTAL','Resultados y validación','CV interna, OOF y test externo se presentan por separado. Todas las cifras proceden de resultados guardados.')
    catalog=st.session_state.catalog
    if catalog is None:
        empty('Resultados no disponibles','Preparar el paquete de artefactos; no se muestran métricas de ejemplo.');return
    root=st.session_state.bundle_root
    comparison,detail,thresholds,external=st.tabs(['Comparación interna','Curvas y folds','Umbrales OOF','Test externo congelado'])
    with thresholds:
        from app_components.thresholds import render as render_thresholds
        render_thresholds(catalog)
    with comparison:
        rows=[]
        for kind,entry in catalog['modalities'].items():
            m=entry['metrics']
            st.subheader(LABELS[kind])
            cards([('AUC CV medio',f'{m["auc_mean"]:.4f}',f'DE muestral: {m["auc_std"]:.4f} · 5 folds'),
                ('AUC OOF',f'{m["oof_auc"]:.4f}','Probabilidades fuera de entrenamiento'),
                ('Desarrollo',str(entry['development_patients']),'Pacientes · una predicción OOF por paciente')])
            for f in entry['folds']:
                rows.append(dict(modelo=LABELS[kind],fold=f['fold'],epocas=f['epochs'],mejor_epoca=f['best_epoch'],
                    **{key:f['metrics'][key] for key in ['auc','accuracy','precision','recall','specificity','balanced_accuracy','f1']}))
        table=pd.DataFrame(rows)
        compact=table[['modelo','fold','auc','accuracy','recall','specificity','balanced_accuracy']].copy()
        compact['modelo']=compact.modelo.replace({LABELS['image']:'Imagen',LABELS['multimodal']:'Multimodal'})
        compact=compact.rename(columns={'modelo':'Modalidad','fold':'Fold','auc':'AUC','accuracy':'Accuracy',
            'recall':'Sensibilidad','specificity':'Especificidad','balanced_accuracy':'Bal. accuracy'})
        st.dataframe(compact,hide_index=True,width='stretch',column_config={
            name:st.column_config.NumberColumn(format='%.4f') for name in ['AUC','Accuracy','Sensibilidad','Especificidad','Bal. accuracy']})
        with st.expander('Tabla completa de métricas y mejores épocas'):
            st.dataframe(table,hide_index=True,width='stretch')
        fig=go.Figure()
        for kind,color in [('image',charts.PURPLE),('multimodal',charts.TEAL)]:
            folds=catalog['modalities'][kind]['folds']
            fig.add_scatter(x=[f['fold'] for f in folds],y=[f['metrics']['auc'] for f in folds],
                mode='lines+markers',name=LABELS[kind],line=dict(color=color))
        charts.style(fig,360,'ROC AUC por paciente · mismos cinco folds')
        fig.update_xaxes(title='Fold de validación',dtick=1);fig.update_yaxes(title='ROC AUC por paciente',range=[0,1])
        plot(fig,'paired_auc')
        st.caption('Comparación descriptiva; cinco folds no son cinco muestras independientes. La selección del checkpoint utiliza el mismo fold de validación: no es CV anidada ni prueba de superioridad clínica.')
        st.download_button('Descargar tabla agregada de folds',table.to_csv(index=False).encode(),file_name='comparacion_folds.csv',mime='text/csv')
    with detail:
        kind=st.selectbox('Modelo a revisar',['multimodal','image'],format_func=lambda k:LABELS[k],key='results_kind')
        entry=catalog['modalities'][kind]
        selected=st.selectbox('Fold',range(5),format_func=lambda f:f'Fold {f}',key='validation_fold')
        f=entry['folds'][selected]
        histories={item['fold']:pd.read_csv(safe_file(root,item['history'])) for item in entry['folds']}
        history=histories[selected]
        st.caption(f"{f['epochs']} épocas completadas · mejor checkpoint: época {f['best_epoch']} por AUC de validación por paciente")
        zoom=st.radio('Escala de pérdidas',['Completa','Zoom automático','Zoom manual'],horizontal=True,key='loss_scale')
        figure,comparable=charts.history_curve(history,'loss')
        figure.add_vline(x=f['best_epoch'],line_dash='dot',line_color=charts.MUTED)
        if zoom=='Zoom automático':
            values=history[['train_eval_loss','val_loss']].to_numpy(dtype=float)
            pad=max(.01,float(np.nanmax(values)-np.nanmin(values))*.08)
            figure.update_yaxes(range=[max(0,float(np.nanmin(values))-pad),float(np.nanmax(values))+pad],rangemode='normal')
            st.caption('Zoom automático: ventana ajustada a los datos completos; no se ocultan épocas ni se suavizan valores.')
        elif zoom=='Zoom manual':
            limit=max(1.,float(history[['train_eval_loss','val_loss']].max().max())*1.1)
            low,high=st.slider('Intervalo visible · BCE',0.,limit,(0.,min(1.,limit)),key=f'loss_ylim_{kind}_{selected}')
            if low<high:figure.update_yaxes(range=[low,high],rangemode='normal')
            st.caption('Solo cambia la ventana visible. No se alteran datos, puntos ni valores fuera del intervalo.')
        plot(figure,'fold_losses')
        st.caption('Train eval y validación: pesos de final de época, sin augmentation. BCE por corte; AUC y accuracy por paciente. No hay suavizado.')
        a,b=st.columns(2)
        for col,metric in [(a,'accuracy'),(b,'auc')]:
            with col:
                curve=charts.history_curve(history,metric)[0]
                curve.add_vline(x=f['best_epoch'],line_dash='dot',line_color=charts.MUTED)
                plot(curve,'fold_'+metric)
        with st.expander('Historial numérico'):
            st.dataframe(history,hide_index=True,width='stretch')
        metric_cards(f['metrics'])
        a,b=st.columns(2)
        with a:plot(roc_plot(entry),'internal_roc')
        with b:plot(charts.confusion_figure(f['metrics']),'fold_confusion')
        st.caption(charts.confusion_legend())
        st.caption('ROC: folds y OOF internos. Matriz: el fold seleccionado; umbral fijo 0,50.')
        st.subheader('Curvas medias de validación cruzada')
        metric=st.selectbox('Métrica que promediar',['loss','auc','accuracy'],
                            format_func=lambda v:{'loss':'BCE por corte','auc':'ROC AUC paciente','accuracy':'Accuracy paciente'}[v],key='mean_metric')
        mode=st.radio('Cobertura de épocas',['Todas las disponibles','Comunes a cinco folds'],horizontal=True,key='mean_mode')
        aggregate=aggregate_histories(histories,'common' if mode.startswith('Comunes') else 'all',
            train_column='train_eval_loss' if metric=='loss' else 'train_eval_patient_'+metric,
            val_column='val_loss' if metric=='loss' else 'val_patient_'+metric)
        if len(aggregate):plot(charts.mean_loss_curve(aggregate,metric=metric),'mean_losses')
        else:st.info('No hay épocas comunes suficientes para esta vista.')
        st.caption('Media ±1 DE muestral. Número de folds explícito por época; sin prolongar curvas tras early stopping ni rellenar épocas.')
        st.dataframe(aggregate,hide_index=True,width='stretch')
        st.download_button('Descargar media por época y cobertura',aggregate.to_csv(index=False).encode(),
                           file_name='media_folds_'+metric+'.csv',mime='text/csv')
    with external:
        entry=catalog['external'];m=entry['metrics']
        st.subheader('Ensemble multimodal · evaluación externa guardada')
        cards([('ROC AUC',f'{m["auc"]:.4f}','Puntuaciones continuas por paciente'),
            ('Pacientes',str(m['n']),'Test externo reservado'),('Umbral','0,50','Fijado antes de consultar test')])
        metric_cards(m)
        if m['recall']==0:
            st.error(f'Sensibilidad cero: {m["tp"]} verdaderos positivos y {m["fn"]} falsos negativos de pCR. La accuracy refleja en gran medida el predominio de no pCR; no demuestra una buena detección de respuesta completa.')
        a,b=st.columns(2)
        with a:plot(roc_plot(entry,True),'external_roc')
        with b:plot(charts.confusion_figure(m),'external_confusion')
        st.caption(charts.confusion_legend())
        st.info('ROC AUC evalúa la ordenación de puntuaciones a distintos umbrales; la matriz usa únicamente 0,50. No se ajusta ese umbral mirando el test.')
        st.caption('Se visualizan estadísticas ya guardadas. Abrir esta página nunca carga imágenes externas, ejecuta el evaluador ni sustituye sus resultados.')
        st.caption('Informe: '+entry['report_id'])
