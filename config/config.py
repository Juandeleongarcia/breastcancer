# =========================================================
# RUTAS Y DATOS
# =========================================================

TRAIN_CSV = "metadata/internal_train.csv"
VAL_CSV = "metadata/internal_val.csv"

DATA_DIR = "breastdcedl"

IMAGE_SIZE = 256
INPUT_CHANNELS = 3

BATCH_SIZE = 16
NUM_WORKERS = 4


# =========================================================
# PREPROCESAMIENTO
# =========================================================

# Modos disponibles:
#
# "raw"
#     canal 0 -> PRE
#     canal 1 -> EARLY
#     canal 2 -> LATE
#
# "enhancement"
#     canal 0 -> PRE
#     canal 1 -> EARLY - PRE
#     canal 2 -> LATE - PRE
#
# Run 005:
INPUT_MODE = "enhancement"


# =========================================================
# CLIPPING DE INTENSIDADES
# =========================================================

# Recorte de valores extremos usando los percentiles
# calculados exclusivamente sobre TRAIN.
#
# Pipeline:
#
# PRE / EARLY / LATE
#       ↓
# [0, 1]
#       ↓
# construcción de canales
#       ↓
# clipping P01-P99
#       ↓
# standardization
#       ↓
# CNN

USE_CLIPPING = True


# Percentil 1 calculado sobre TRAIN
CHANNEL_P01 = [
    0.0000000000,      # PRE
    -0.1098039150,     # EARLY - PRE
    -0.1058823615,     # LATE - PRE
]


# Percentil 99 calculado sobre TRAIN
CHANNEL_P99 = [
    0.6705882549,      # PRE
    0.6509803534,      # EARLY - PRE
    0.6235294342,      # LATE - PRE
]


# =========================================================
# ESTANDARIZACIÓN
# =========================================================

# Estandarización independiente por canal:
#
#     x_standardized = (x - mean) / std
#
# Las estadísticas se han calculado sobre TRAIN
# DESPUÉS de aplicar el clipping.
USE_STANDARDIZATION = True


CHANNEL_MEANS = [
    0.1263833838,
    0.0676578613,
    0.0782202765,
]


CHANNEL_STDS = [
    0.1545330693,
    0.1365353164,
    0.1387470868,
]


# =========================================================
# ARQUITECTURA CNN
# =========================================================

# Arquitectura compacta mantenida respecto al Run 004
# para aislar el efecto del preprocessing.
CONV_CHANNELS = [
    32,
    64,
    128,
]

KERNEL_SIZE = 3
PADDING = 1

POOL_SIZE = 2

USE_BATCH_NORM = True

DROPOUT = 0.40

# Clasificación binaria:
# 1 -> pCR
# 0 -> no pCR
NUM_CLASSES = 1


# =========================================================
# ENTRENAMIENTO
# =========================================================

EPOCHS = 25

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005


# Opciones implementadas:
# "adam"
# "adamw"
# "sgd"
OPTIMIZER = "adamw"


# Mantenemos la misma weighted loss que en Run 004
# para que el cambio principal de Run 005
# sea exclusivamente el preprocessing.
USE_WEIGHTED_LOSS = True


# =========================================================
# CLASIFICACIÓN
# =========================================================

# Threshold utilizado para:
# - Accuracy
# - Sensitivity
# - Specificity
#
# No afecta al ROC-AUC.
CLASSIFICATION_THRESHOLD = 0.50


# =========================================================
# EARLY STOPPING
# =========================================================

USE_EARLY_STOPPING = True

# Detiene el entrenamiento si el validation ROC-AUC
# no mejora durante 10 épocas consecutivas.
PATIENCE = 10


# =========================================================
# MÉTRICAS
# =========================================================

# Objetivo marcado para el proyecto.
TARGET_ROC_AUC = 0.70


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

SEED = 42


# =========================================================
# HARDWARE
# =========================================================

# Si la GPU está disponible se utilizará automáticamente.
USE_GPU = True

# Opciones nuevas; los valores originales del baseline se conservan arriba.
# None conserva PATIENCE/SEED como fuente para los nombres históricos.
EARLY_STOPPING_PATIENCE = None
RANDOM_SEED = None
DEVICE = "auto"  # auto, cpu o cuda; auto respeta USE_GPU
TORCH_NUM_THREADS = 4
USE_AUGMENTATION = False  # volteos geométricos conjuntos, solo train
SMOKE_TEST = False
SMOKE_PATIENTS_PER_CLASS = 2
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None
RUN_NAME = None  # sufijo descriptivo; el identificador siempre es único
RESULTS_DIR = "models"
SAVE_EVERY = 1
PRIMARY_METRIC = "patient_auc"
STATS_PATH = None
STATS_PROVENANCE = "legacy_unverified"  # constantes conservadas del Run 005
STATS_PREPROCESSING = {
    "INPUT_MODE": "enhancement",
    "USE_CLIPPING": True,
    "CHANNEL_P01": [0.0, -0.1098039150, -0.1058823615],
    "CHANNEL_P99": [0.6705882549, 0.6509803534, 0.6235294342],
}

# Fase 4: valores neutros que mantienen exactamente la arquitectura y dinámica originales.
CONVS_PER_BLOCK = 1
SCHEDULER = "none"  # none o plateau (AUC de paciente, mode=max)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 0.000001

# Fase 5: la opción histórica conserva sus volteos; el nuevo protocolo es explícito.
AUGMENTATION_MODE = "legacy_flips"
AUGMENT_PROBABILITY = 0.8
AUGMENT_DEGREES = 5.0
AUGMENT_TRANSLATE = 0.02
AUGMENT_SCALE_DELTA = 0.02
