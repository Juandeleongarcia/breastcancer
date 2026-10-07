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
# Para continuar desde el Run 003:
INPUT_MODE = "enhancement"


# ---------------------------------------------------------
# ESTANDARIZACIÓN
# ---------------------------------------------------------

# Si es True:
#
#     canal = (canal - mean) / std
#
# IMPORTANTE:
# Las estadísticas deben calcularse únicamente
# utilizando el conjunto de entrenamiento.
#
# Por ahora lo dejamos desactivado hasta calcularlas.
USE_STANDARDIZATION = False


# Valores utilizados únicamente si
# USE_STANDARDIZATION = True.
#
# De momento son neutros:
CHANNEL_MEANS = [
    0.0,
    0.0,
    0.0,
]

CHANNEL_STDS = [
    1.0,
    1.0,
    1.0,
]


# =========================================================
# ARQUITECTURA CNN
# =========================================================

# Run 001:
# [32, 64, 128]
# Best Val AUC ~ 0.5656
#
# Run 002:
# [32, 64, 128, 256]
# Best Val AUC ~ 0.5452
#
# Run 003:
# [32, 64, 128]
# + enhancement
# Best Val AUC ~ 0.5596
#
# La arquitectura compacta ha generalizado mejor.
CONV_CHANNELS = [
    32,
    64,
    128,
]

KERNEL_SIZE = 3

PADDING = 1

# MaxPool2d(2)
POOL_SIZE = 2

USE_BATCH_NORM = True

DROPOUT = 0.40

# Clasificación binaria:
# pCR / no pCR
NUM_CLASSES = 1


# =========================================================
# ENTRENAMIENTO
# =========================================================

# Aumentamos el máximo porque ahora permitiremos
# más épocas sin mejora.
EPOCHS = 25

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

# Opciones implementadas:
# "adam"
# "adamw"
# "sgd"
OPTIMIZER = "adamw"

# Compensación del desbalance entre
# pCR y no pCR.
USE_WEIGHTED_LOSS = True


# =========================================================
# CLASIFICACIÓN
# =========================================================

# Afecta a:
# - accuracy
# - sensitivity
# - specificity
#
# NO afecta al ROC-AUC.
CLASSIFICATION_THRESHOLD = 0.50


# =========================================================
# EARLY STOPPING
# =========================================================

USE_EARLY_STOPPING = True

# Antes utilizábamos 5.
#
# Ahora permitimos 10 épocas consecutivas
# sin mejora del validation ROC-AUC.
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