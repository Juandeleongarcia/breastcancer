"""Figuras interactivas a partir de datos existentes, sin suavizado ni relleno."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score

TEAL = '#42D3C4'
WHITE = '#DFE8F3'
MUTED = '#8CA2BD'
PURPLE = '#B5A9EE'
AMBER = '#D9B878'

CNN_CONCEPTS = (
    ('Imágenes','Un estudio aporta cortes e información temporal que deben mantenerse correctamente asociados.'),
    ('Patrones locales','Las convoluciones aprenden filtros espaciales a partir de los ejemplos de entrenamiento.'),
    ('Transformaciones','La normalización y las activaciones ayudan a combinar los mapas de características.'),
    ('Contexto','La reducción espacial permite reunir información de regiones más amplias.'),
    ('Representación','Los mapas se resumen y regularizan según la arquitectura que se seleccione.'),
    ('Estimación','El clasificador producirá una puntuación; su inferencia y configuración siguen pendientes.')
)


def conceptual_cnn_figure(selected):
    count = len(CNN_CONCEPTS)
    figure = go.Figure(go.Scatter(x=list(range(count)),y=[0]*count,mode='lines+markers+text',
        text=[label for label,_ in CNN_CONCEPTS],textposition='bottom center',textfont=dict(size=12),
        line=dict(color='#2B4C60',width=3),
        marker=dict(size=[26 if i==selected else 18 for i in range(count)],
                    color=[TEAL if i==selected else '#47617B' for i in range(count)]),
        customdata=[description for _,description in CNN_CONCEPTS],
        hovertemplate='%{text}<br>%{customdata}<extra></extra>'))
    style(figure,235)
    figure.update_layout(showlegend=False,hovermode='closest',margin=dict(l=45,r=45,t=25,b=45))
    figure.update_xaxes(visible=False,range=[-.4,count-.6])
    figure.update_yaxes(visible=False,range=[-.7,.55])
    return figure


def style(figure, height=340, title=None):
    figure.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Segoe UI, Arial, sans-serif',size=13,color=WHITE),
        margin=dict(l=45,r=25,t=65 if title else 24,b=105),height=height,
        title=dict(text=title,font=dict(size=15)) if title else None,
        legend=dict(orientation='h',y=-.24,x=0,yanchor='top',font=dict(size=11)),
        hoverlabel=dict(bgcolor='#15283B'),hovermode='x unified')
    figure.update_xaxes(showgrid=False,zeroline=False,automargin=True)
    figure.update_yaxes(gridcolor='#203146',zeroline=False,automargin=True)
    return figure


def history_curve(history, metric):
    figure = go.Figure()
    if metric == 'loss':
        training = 'train_eval_loss' if 'train_eval_loss' in history else 'train_loss'
        validation = 'val_loss'; ylabel = 'BCE / corte'
    else:
        training = f'train_eval_patient_{metric}' if f'train_eval_patient_{metric}' in history else f'train_patient_{metric}'
        validation = f'val_patient_{metric}'; ylabel = 'ROC AUC paciente' if metric=='auc' else 'Accuracy paciente'
    comparable = training.startswith('train_eval')
    if training in history:
        figure.add_scatter(x=history.epoch,y=history[training],name='Train eval' if comparable else 'Train online · pesos variables',
                           mode='lines+markers',marker=dict(size=5),line=dict(color=TEAL if comparable else MUTED,dash='solid' if comparable else 'dash'))
    if validation in history:
        figure.add_scatter(x=history.epoch,y=history[validation],name='Validación',mode='lines+markers',marker=dict(size=5),line=dict(color=PURPLE,dash='dash'))
    style(figure,390)
    figure.update_xaxes(title='Época',dtick=1)
    values=np.concatenate([np.asarray(t.y,dtype=float) for t in figure.data]) if figure.data else np.array([0.])
    finite=values[np.isfinite(values)]
    upper=max(1.,float(finite.max())*1.08) if finite.size and metric=='loss' else 1.
    figure.update_yaxes(title=ylabel,range=[0,upper],rangemode='tozero')
    return figure, comparable


def mean_loss_curve(table, training_label='Train eval',metric='loss'):
    figure = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.79,.21],vertical_spacing=.1)
    for name,label,color,fill in [('train',training_label,TEAL,'rgba(66,211,196,.12)'),
                                  ('val','Validación',PURPLE,'rgba(181,169,238,.12)')]:
        lower = table[name+'_mean']-table[name+'_std']
        upper = table[name+'_mean']+table[name+'_std']
        figure.add_scatter(x=table.epoch,y=lower,mode='lines',line=dict(width=0),showlegend=False,
                           hoverinfo='skip',connectgaps=False,row=1,col=1)
        figure.add_scatter(x=table.epoch,y=upper,mode='lines',line=dict(width=0),fill='tonexty',fillcolor=fill,
                           showlegend=False,hoverinfo='skip',connectgaps=False,row=1,col=1)
        figure.add_scatter(x=table.epoch,y=table[name+'_mean'],mode='lines+markers',name=label,marker=dict(size=5),
                           line=dict(color=color,dash='dash' if name=='val' else 'solid'),customdata=table.n_folds,
                           hovertemplate='%{y:.5f} · %{customdata} folds<extra>'+label+'</extra>',row=1,col=1)
    figure.add_bar(x=table.epoch,y=table.n_folds,name='Folds participantes',marker_color='#36526B',
                   showlegend=False,row=2,col=1,hovertemplate='Época %{x}: %{y} folds<extra></extra>')
    style(figure,height=480)
    values=np.concatenate([(table[n+'_mean']+table[n+'_std'].fillna(0)).to_numpy() for n in ['train','val']])
    upper=max(1.,float(values.max())*1.08) if metric=='loss' else 1.
    figure.update_yaxes(title={'loss':'BCE / corte','auc':'ROC AUC paciente','accuracy':'Accuracy paciente'}[metric],
                       range=[0,upper],row=1,col=1)
    figure.update_yaxes(title='Folds',dtick=1,range=[0,5.5],row=2,col=1)
    figure.update_xaxes(title='Época',dtick=1,row=2,col=1)
    return figure


def roc_figure(folds, oof=None):
    figure = go.Figure()
    colors = [TEAL,PURPLE,'#85ABD4','#D9B878','#C999AF']
    for fold in folds:
        if fold.predictions.empty: continue
        data = fold.predictions
        x,y,_ = roc_curve(data.label,data.probability)
        figure.add_scatter(x=x,y=y,name=f'Fold {fold.fold} · AUC {roc_auc_score(data.label,data.probability):.4f}',
                           mode='lines',line=dict(color=colors[fold.fold],width=1.5))
    if oof is not None and not oof.empty:
        x,y,_ = roc_curve(oof.label,oof.probability)
        figure.add_scatter(x=x,y=y,name=f'OOF · AUC {roc_auc_score(oof.label,oof.probability):.4f}',
                           mode='lines',line=dict(color=WHITE,width=3))
    figure.add_scatter(x=[0,1],y=[0,1],name='Referencia aleatoria',mode='lines',line=dict(color=MUTED,dash='dot'))
    style(figure,410)
    figure.update_layout(hovermode='closest',legend=dict(orientation='h',y=-.24))
    figure.update_xaxes(title='1 − especificidad',range=[0,1])
    figure.update_yaxes(title='Sensibilidad',range=[0,1])
    return figure


def confusion_figure(metrics):
    matrix = metrics['confusion_matrix']
    text = [[f'TN · {matrix[0][0]}',f'FP · {matrix[0][1]}'],[f'FN · {matrix[1][0]}',f'TP · {matrix[1][1]}']]
    figure = go.Figure(go.Heatmap(z=[[1,0],[0,1]],x=['no pCR','pCR'],y=['no pCR','pCR'],customdata=matrix,
        text=text,texttemplate='%{text}',textfont=dict(size=18,color=WHITE),colorscale=[[0,'#653E50'],[1,'#15574F']],
        zmin=0,zmax=1,xgap=4,ygap=4,
        showscale=False,hovertemplate='Real: %{y}<br>Predicción: %{x}<br>Pacientes: %{customdata:d}<extra></extra>'))
    style(figure,350)
    figure.update_layout(hovermode='closest',margin=dict(l=65,r=20,t=15,b=65))
    figure.update_xaxes(title='Predicción por paciente')
    figure.update_yaxes(title='Etiqueta real',autorange='reversed')
    return figure


def confusion_legend():
    return ('TN: predice no pCR y es no pCR · FP: predice pCR y es no pCR · '
            'FN: predice no pCR y es pCR · TP: predice pCR y es pCR. '
            'Turquesa: aciertos; coral: errores. El color distingue el tipo de resultado, no su frecuencia.')


def fold_probabilities(result):
    values=np.asarray(result['member_probabilities'],dtype=float)
    if values.shape!=(5,) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Se requieren cinco probabilidades válidas de los folds')
    if not np.isclose(values.mean(),result['patient_probability'],rtol=0,atol=1e-12):
        raise ValueError('La media del paciente no coincide con los cinco miembros')
    labels=[f'Fold {fold}' for fold in range(5)]
    fig=go.Figure(go.Scatter(x=labels,y=values,mode='markers',name='Cada modelo',
        marker=dict(size=13,color=TEAL,line=dict(color=WHITE,width=1)),
        hovertemplate='%{x}<br>Puntuación pCR: %{y:.6f}<extra></extra>'))
    for value,label,color,dash in [(result['patient_probability'],'Media paciente',PURPLE,'solid'),
                                   (result['threshold'],'Umbral',AMBER,'dash')]:
        fig.add_scatter(x=labels,y=[value]*5,mode='lines',name=f'{label} · {value:.4f}',
                        line=dict(color=color,dash=dash),hovertemplate=label+': %{y:.6f}<extra></extra>')
    style(fig,400,'Cinco modelos · media y umbral de la muestra')
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(title='Modelo del fold',type='category',categoryorder='array',categoryarray=labels)
    fig.update_yaxes(title='Puntuación pCR',range=[0,1])
    return fig


def threshold_curves(report,selected,youden=False):
    fig=go.Figure()
    series=[('youden','Youden',TEAL)] if youden else [
        ('sensitivity','Sensibilidad',TEAL),('specificity','Especificidad',PURPLE),
        ('precision','Precisión pCR',AMBER),('f1','F1','#85ABD4'),('balanced_accuracy','Balanced accuracy',WHITE)]
    for metric,label,color in series:
        fig.add_scatter(x=[r['threshold'] for r in report['thresholds']],y=[r[metric] for r in report['thresholds']],
            name=label,mode='lines',line=dict(color=color),hovertemplate=label+': %{y:.4f}<extra></extra>')
    # Referencias como trazas: leyenda debajo, sin anotaciones que tapen las curvas.
    limits=[-1,1] if youden else [0,1]
    for value,label,color in [(report['recommended_threshold'],'Exploratorio OOF',TEAL),(.5,'Original',AMBER)]:
        fig.add_scatter(x=[value,value],y=limits,mode='lines',name=f'{label} {value:.2f}',
                       line=dict(color=color,dash='dash'),hoverinfo='skip')
    if selected not in [report['recommended_threshold'],.5]:
        fig.add_scatter(x=[selected,selected],y=limits,mode='lines',name=f'Explorado {selected:.2f}',
                       line=dict(color=MUTED,dash='dot'),hoverinfo='skip')
    style(fig,490,'Índice de Youden · desarrollo OOF' if youden else 'Sensibilidad y métricas · desarrollo OOF')
    fig.update_xaxes(title='Umbral pCR',range=[.05,.95])
    fig.update_yaxes(title='Youden = sensibilidad + especificidad − 1' if youden else 'Proporción',range=limits)
    return fig


def fold_auc_figure(report):
    scores = [f.metrics.get('auc') if f.complete else None for f in report.folds]
    figure = go.Figure(go.Bar(x=[f'Fold {i}' for i in range(5)],y=scores,marker_color=TEAL,
                             text=[f'{v:.4f}' if v is not None else '' for v in scores],textposition='outside'))
    if report.completed:
        mean = report.metrics['auc_mean']
        figure.add_hline(y=mean,line_dash='dash',line_color=PURPLE,
                        annotation_text='Media' if report.complete else 'Media provisional')
    for i,value in enumerate(scores):
        if value is None:figure.add_annotation(x=f'Fold {i}',y=.06,text='Pendiente',showarrow=False,font=dict(color=MUTED,size=12))
    style(figure)
    figure.update_yaxes(title='ROC AUC paciente',range=[0,1])
    return figure


def lr_figure(history):
    figure = go.Figure(go.Scatter(x=history.epoch,y=history.learning_rate,mode='lines+markers',line=dict(color=AMBER)))
    style(figure,250)
    figure.update_xaxes(title='Época',dtick=1)
    figure.update_yaxes(title='LR utilizado',type='log')
    return figure


def slice_probabilities(result):
    frame = result['slices']
    figure = go.Figure(go.Scatter(x=frame.slice_index,y=frame.probability,mode='lines+markers',
                    line=dict(color=TEAL),marker=dict(size=8),name='pCR por corte'))
    figure.add_hline(y=result['patient_probability'],line_color=PURPLE,
                    annotation_text='Media del paciente',line_dash='dash')
    figure.add_hline(y=result['threshold'],line_color=MUTED,line_dash='dot',annotation_text='Umbral guardado')
    style(figure,320)
    figure.update_xaxes(title='Índice original del corte')
    figure.update_yaxes(title='Puntuación estimada de pCR',range=[0,1])
    return figure


def probability_donut(probability):
    figure = go.Figure(go.Pie(values=[probability,1-probability],labels=['pCR','no pCR'],hole=.82,
        marker=dict(colors=[TEAL,'#25364D'],line=dict(width=0)),textinfo='none',sort=False,
        hovertemplate='%{label}: %{percent}<extra></extra>'))
    style(figure,275)
    figure.update_layout(showlegend=False,margin=dict(l=10,r=10,t=10,b=10),
        annotations=[dict(text=f'{probability:.1%}',x=.5,y=.53,showarrow=False,font=dict(size=35,color=WHITE)),
                     dict(text='pCR estimada',x=.5,y=.32,showarrow=False,font=dict(size=13,color=MUTED))])
    return figure


def slice_stack(study, phase, selected):
    count = len(study.slice_indices)
    indices = sorted(set(np.linspace(0,count-1,min(12,count)).astype(int).tolist()+[selected]))
    figure = go.Figure()
    axis = np.arange(0,256,4)
    x,y = np.meshgrid(axis,axis)
    for i in indices:
        figure.add_surface(x=x,y=y,z=np.full(x.shape,study.slice_indices[i]),surfacecolor=study.phases[i,phase,::4,::4]/255.,
            cmin=0,cmax=1,colorscale='Gray',opacity=.95 if i==selected else .20,showscale=False,
            name=f'z{study.slice_indices[i]:03d}',hovertemplate=f'Corte z{study.slice_indices[i]:03d}<extra></extra>')
    figure.add_scatter3d(x=[0,255,255,0,0],y=[0,0,255,255,0],z=[study.slice_indices[selected]]*5,
                         mode='lines',line=dict(color=TEAL,width=5),showlegend=False,hoverinfo='skip')
    style(figure,510,'Pila esquemática · no reconstrucción anatómica')
    figure.update_layout(hovermode='closest',scene=dict(bgcolor='rgba(0,0,0,0)',
        xaxis_title='X (píxel)',yaxis_title='Y (píxel)',zaxis_title='Índice axial z · sin mm',
        aspectmode='manual',aspectratio=dict(x=1,y=1,z=.7),camera=dict(eye=dict(x=1.5,y=-1.7,z=1.1))))
    return figure, len(indices)


def intensity_surface(study,phase,selected=0):
    axis=np.arange(0,256,4);x,y=np.meshgrid(axis,axis)
    figure=go.Figure(go.Surface(x=x,y=y,z=study.phases[selected,phase,::4,::4]/255.,
        colorscale='Viridis',colorbar=dict(title='Intensidad'),hovertemplate='x=%{x}, y=%{y}<br>Intensidad=%{z:.3f}<extra></extra>'))
    style(figure,500,'Superficie matemática de intensidad · no reconstrucción anatómica')
    figure.update_layout(scene=dict(xaxis_title='X (píxel)',yaxis_title='Y (píxel)',zaxis_title='Intensidad PNG [0,1]'))
    return figure


def temporal_curve(study,selected,x,y,radius):
    region=study.phases[selected,:,max(0,y-radius):min(256,y+radius+1),max(0,x-radius):min(256,x+radius+1)]
    values=region.astype(float).mean(axis=(1,2))/255.
    figure=go.Figure(go.Scatter(x=['PRE','EARLY','LATE'],y=values,mode='lines+markers',
        line=dict(color=TEAL,width=3),marker=dict(size=10),name='Media de la región'))
    style(figure,300,'Tres adquisiciones temporales · intensidad media de la región')
    figure.update_xaxes(title='Fase temporal · sin tiempos de adquisición en segundos')
    figure.update_yaxes(title='Intensidad normalizada PNG / 255 [0,1]',range=[0,1])
    return figure


def image_zoom(image, title, signed=False):
    figure = go.Figure(go.Heatmap(z=image,colorscale='RdBu' if signed else 'Gray',
                                  showscale=True,colorbar=dict(title='Valor'),hovertemplate='x=%{x}, y=%{y}<br>%{z:.4f}<extra></extra>'))
    style(figure,570,title)
    figure.update_xaxes(title='Píxel x')
    figure.update_yaxes(title='Píxel y',autorange='reversed',scaleanchor='x')
    return figure
