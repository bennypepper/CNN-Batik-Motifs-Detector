import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, 'models', 'batik_model_v2.tflite')
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image for the TFLite model.

    The training pipeline applied preprocess_input() externally (scaling
    [0, 255] → [-1, 1]) before feeding into EfficientNetV2S which also has
    The TFLite model contains an internal rescaling_1_1 layer
    (include_preprocessing=True in the rebuilt graph) that maps [0, 255] → [-1, 1].
    Diagnostic confirmed: raw [0-255] input produces strongest signal.
    Do NOT apply external preprocess_input() — let the model handle it.
    """
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return np.expand_dims(image_array.astype('float32'), axis=0)
