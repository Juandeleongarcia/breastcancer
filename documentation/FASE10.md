# Fase 10: auditoría y control manual de campañas

Guía operativa: [GPU_FASE10.md](../GPU_FASE10.md).

## Alcance y evidencia local

La campaña `CV9_20261028_183759_SPATIAL_SMALL_531c826715d2` existe únicamente en el
PC de clase. No se han podido inspeccionar sus checkpoints desde Windows ni se
declara recuperado su fold 2. El controlador consulta sus resultados reales al
ejecutarse allí; nunca prepara otra campaña de control si falta esa ruta.

No se han realizado entrenamientos clínicos ni creado campañas simuladas dentro
de `models/`. Las pruebas de orquestación usan temporales y procesos simulados.
No se han cambiado el dataset, los CSV, configuraciones históricas, arquitecturas,
preprocesamiento ni RUNs existentes. Hay una copia de seguridad previa local
bajo `backups/`, excluida de Git.

## Problemas encontrados y adaptación mínima

1. Fase 9 guarda hashes de trece archivos del ejecutor en cada campaña. Cambiarlos
   impediría recuperar la campaña existente sin una migración. Se conservan todos
   y se añade únicamente `src.phase10` como controlador externo.
2. El CLI de Fase 9 solo admite `SPATIAL_SMALL` y `SPATIAL_CONTROL`. Los candidatos
   reales sí existen en `config/phase8/F8_SMALL_DROPOUT.json` y
   `config/phase8/F8_SMALL_LR.json`. Se reutilizan mediante el preparador F9 y se
   identifica cada campaña nueva antes de ajustar o entrenar sus folds.
3. F8 apunta a estadísticas de una partición interna única. No sirven directamente
   para cinco folds: se mantienen los operadores de preprocesamiento, pero se usa
   el ajuste F9 por fold exclusivamente en sus pacientes de entrenamiento.
4. El ejecutor F9 puede continuar con otros folds dentro del mismo proceso tras
   una excepción. Para la recuperación GPU F10 utiliza un hijo Python por fold y
   se detiene ante el primer fallo, conservando logs y checkpoints. No hay reintentos
   automáticos ni cambios de drivers o entorno.
5. La selección de checkpoints F10 conserva y descarta archivos ilegibles o
   incompatibles, registra el motivo y busca un límite de época válido. Reiniciar
   desde cero requiere `--restart-fold`; no ocurre silenciosamente.

## Contrastes exactos

El controlador comprueba que F8 difiere del control SMALL F7 únicamente en dropout
o LR, aparte de la identificación. Se conservan AdamW, weight decay 0,0005, batch
16, semilla 42, veinte épocas, paciencia ocho, BatchNorm, scheduler plateau,
loss no ponderada, ausencia de augmentation y evaluación completa de train.
Se adaptan las rutas/estadísticas/recuentos derivados y los metadatos descriptivos
necesarios para CV, sin cambiar los hiperparámetros.

Los cinco folds oficiales proceden de `metadata/samples.csv`. F9 valida pacientes,
etiquetas, CSV derivados, estadísticas y cobertura OOF. El fold reservado -1 queda
excluido. El mismo inicializador con semilla 42 crea un modelo independiente por
fold; ninguna variante recibe pesos de SMALL.

## Recuperación, concurrencia y resultados

El manifiesto conserva un historial `phase10_attempts` con modo, checkpoint,
rechazos y error previo. El reinicio explícito deja constancia incluso de un
checkpoint compatible que se decidió no utilizar. Los RUNs terminados se omiten
solo tras comprobar sus artefactos, incluso si el estado del manifiesto quedó
desactualizado tras una interrupción.

El registro persistente `models/phase10/campaigns.json` y la adopción de una única
campaña propia impiden duplicar B/C al repetir comandos. Un bloqueo Linux `flock`
excluye otros controladores F10 y una inspección de `/proc` detecta ejecutores
anteriores activos. Los ejecutores antiguos no adquieren este bloqueo: no deben
lanzarse en paralelo manualmente.

Las comparaciones se escriben con fecha/hora/UUID. Solo se publica media final,
desviación estándar y OOF cuando los cinco folds están completos. Se usan siempre
resultados CV de desarrollo, sin mezclarlos con holdouts históricos. Los resultados
y logs quedan bajo `models/`, ya excluido de Git.

## Comprobación reproducible

```text
python -m src.phase10 configs
python -m src.phase10 --help
python -m src.cross_validation --help
python -B -m unittest discover -s tests -p test_phase10.py -v
python -B -m unittest discover -s tests -v
```

La comprobación real de HIP/GPU y la recuperación de los checkpoints de clase
quedan pendientes para ese equipo. No se han inventado métricas nuevas.
