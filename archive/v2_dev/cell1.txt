import subprocess
subprocess.run(["pip", "install", "-q", "albumentations==1.4.3"], check=True)

import os, random, warnings, platform, datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import seaborn as sns, cv2

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import TopKCategoricalAccuracy

import albumentations as A

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = "/kaggle/input/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned/dataset"
OUTPUT_DIR  = "/kaggle/working"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_batik_model_v2.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "batik_model_v2.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")

IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_SIZE      = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS  = 3
INPUT_SHAPE   = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE         = 32
N_FOLDS            = 5
PHASE1_EPOCHS      = 20
PHASE2_EPOCHS      = 40
PHASE1_LR          = 1e-3
PHASE2_LR          = 1e-5
PHASE1_PATIENCE    = 5
PHASE2_PATIENCE    = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR   = 0.3
MIN_LR             = 1e-7
UNFREEZE_LAYERS    = 40
DROPOUT_RATE_1     = 0.5
DROPOUT_RATE_2     = 0.3
DENSE_UNITS        = 256
LABEL_SMOOTHING    = 0.1
TTA_STEPS          = 5
TEST_SIZE          = 0.15
TIER1_THRESHOLD    = 45
TIER2_THRESHOLD    = 90

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU devices found: {len(gpus)}")
else:
    print("WARNING: No GPU detected. Training will run on CPU.")

tf.keras.mixed_precision.set_global_policy("mixed_float16")
print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")

print("\nEnvironment Configuration:")
print(f"  Platform        : {platform.platform()}")
print(f"  Python          : {platform.python_version()}")
print(f"  TensorFlow      : {tf.__version__}")
print(f"  Backbone        : EfficientNetV2S")
print(f"  Dataset path    : {DATASET_DIR}")
print(f"  Image size      : {IMG_SIZE}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  K-Folds         : {N_FOLDS}")
print(f"  Label smoothing : {LABEL_SMOOTHING}")
print(f"  TTA steps       : {TTA_STEPS}")
print(f"  Test holdout    : {int(TEST_SIZE * 100)}%")