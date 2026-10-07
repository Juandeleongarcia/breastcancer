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
# ARQUITECTURA CNN
# =========================================================

# Número de filtros de cada bloque convolucional.
# Con [32, 64, 128] se crean 3 bloques convolucionales.
CONV_CHANNELS = [32, 64, 128]

KERNEL_SIZE = 3
PADDING = 1

# MaxPool2d(2) reduce a la mitad alto y ancho.
POOL_SIZE = 2

USE_BATCH_NORM = True

DROPOUT = 0.30

# Clasificación binaria: pCR / no pCR
NUM_CLASSES = 1


# =========================================================
# ENTRENAMIENTO
# =========================================================

# Primera prueba seria en GPU
EPOCHS = 10

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

# Opciones implementadas:
# "adam", "adamw", "sgd"
OPTIMIZER = "adam"

# Compensa el desbalance entre pCR y no pCR
USE_WEIGHTED_LOSS = True


# =========================================================
# CLASIFICACIÓN
# =========================================================

# Probabilidad mínima para clasificar una muestra como positiva.
CLASSIFICATION_THRESHOLD = 0.50


# =========================================================
# EARLY STOPPING
# =========================================================

USE_EARLY_STOPPING = True

# Detiene el entrenamiento si el AUC no mejora
# durante este número de épocas consecutivas.
PATIENCE = 4


# =========================================================
# GUARDADO DEL MODELO
# =========================================================

MODEL_PATH = "models/best_model.pt"

SAVE_BEST_MODEL = True


# =========================================================
# MÉTRICAS
# =========================================================

# Objetivo marcado para el proyecto
TARGET_ROC_AUC = 0.70


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

SEED = 42


# =========================================================
# HARDWARE
# =========================================================

# Si CUDA está disponible, train.py utilizará la GPU.
# Si no, utilizará CPU automáticamente.
USE_GPU = True