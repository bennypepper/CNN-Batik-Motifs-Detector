import os
import tensorflow as tf
import gc

print("\nRebuilding model in float32 to bypass mixed-precision TFLite trace deadlock...")
tf.keras.mixed_precision.set_global_policy("float32")

# Rebuild EfficientNetV2S architecture in pure float32 to remove dynamic Cast ops
inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
base_model_fp32 = tf.keras.applications.EfficientNetV2S(weights=None, include_top=False, input_tensor=inputs)
x = base_model_fp32.output
x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
x = tf.keras.layers.Dropout(DROPOUT_RATE_1, name="dropout_1")(x)
x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_head")(x)
x = tf.keras.layers.Dropout(DROPOUT_RATE_2, name="dropout_2")(x)
x = tf.keras.layers.Dense(NUM_CLASSES, name="dense_logits")(x)
outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
model_fp32 = tf.keras.models.Model(inputs=inputs, outputs=outputs)

print("Loading trained weights into float32 graph...")
model_fp32.load_weights(MODEL_PATH)

print("Freeing GPU memory to prevent TFLiteConverter hang...")
if "best_model" in dir():
    del best_model
tf.keras.backend.clear_session()
gc.collect()

print("Converting to TFLite (FP16) on CPU directly from Keras model...")
with tf.device("/CPU:0"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

keras_size  = os.path.getsize(MODEL_PATH) / (1024 * 1024)
tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print(f"\nKeras model size : {keras_size:.1f} MB")
print(f"TFLite size      : {tflite_size:.1f} MB")
print("V2 Pipeline Complete. TFLite model is ready for deployment.")
