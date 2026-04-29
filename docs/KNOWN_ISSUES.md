# Known Issues & Technical Notes — v2.0

This document records known issues, root-cause analyses, and workarounds discovered during v2 development and deployment. It serves as a reference for contributors and as a backlog for future versions.

---

## Table of Contents

1. [Critical: Double-Preprocessing Bug](#1-critical-double-preprocessing-bug)
2. [Weak Classes (F1 < 0.65)](#2-weak-classes-f1--065)
3. [TFLite Conversion Workaround](#3-tflite-conversion-workaround)
4. [Recommendations for v3](#4-recommendations-for-v3)

---

## 1. Critical: Double-Preprocessing Bug

### Status: **Workaround Applied** (v2.0) — Proper fix requires retraining (v3)

### Summary

The v2 training pipeline accidentally applied image preprocessing **twice** before inference — once externally via `preprocess_input()` and once internally via the model's built-in `Rescaling` layer. This caused the model to train on a severely compressed input distribution, limiting its discriminative power and directly causing wrong predictions during deployment when the preprocessing wasn't matched correctly.

### Root Cause

The training pipeline (cell3) applies `preprocess_input()` from `tensorflow.keras.applications.efficientnet_v2` as the last step before feeding images to the model:

```python
# cell3.txt, line 102 — inside the _wrap() function
return preprocess_input(img), tf.one_hot(lbl, depth=NUM_CLASSES)
```

`preprocess_input()` for EfficientNetV2 scales pixel values: `(x / 127.5) - 1.0`, mapping `[0, 255] → [-1, 1]`.

However, `EfficientNetV2S` was instantiated with `include_preprocessing=True` (the default), which bakes a `Rescaling` layer into the model graph itself:

```python
# cell7.txt, line 10 — TFLite graph rebuild
base_model_fp32 = tf.keras.applications.EfficientNetV2S(
    weights=None, include_top=False, input_tensor=inputs
)
# include_preprocessing defaults to True → internal Rescaling layer included
```

This means during training, the data path was:

```
Raw image [0, 255]
    → external preprocess_input()  →  [-1, 1]
    → model's internal Rescaling   →  [-1.008, -0.992]   ← collapsed range!
    → EfficientNetV2S backbone
    → classification head
```

The model trained on pixel values compressed to **~1.6% of the intended dynamic range** (a band of ~0.016 around -1.0 instead of the full 2.0 range from -1 to +1).

### How We Discovered It

After deploying the TFLite model in the Streamlit app, predictions were consistently wrong regardless of which preprocessing we applied. We ran a diagnostic script that inspected the TFLite model's internal tensor structure:

```
=== MODEL LAYER INSPECTION ===
  functional_1/rescaling_1_1/Cast/x      shape=[]            ← Rescaling layer EXISTS
  functional_1/rescaling_1_1/Cast_1      shape=[]
  functional_1/rescaling_1_1/mul         shape=[1,224,224,3]
  functional_1/rescaling_1_1/add         shape=[1,224,224,3]
```

Confirmed: the TFLite model **does** contain an internal `rescaling_1_1` layer.

We then tested inference with different input ranges:

| Input Type | Value Range | Max Confidence | Std Dev | Verdict |
|---|---|---|---|---|
| `raw [0-255]` | 0 to 255 | **0.194** | **0.042** | **Best signal** |
| `scaled [-1,1]` | -1 to 1 | 0.100 | 0.028 | Weak |
| `double-scaled` | ≈ -1.008 to -0.992 | 0.070 | 0.016 | Garbage (≈ uniform) |
| `zeros` | all 0 | 0.070 | 0.016 | Garbage baseline |

The `raw [0-255]` input produced the strongest predictions because it passes through the model's internal Rescaling layer to get proper `[-1, 1]` values — exactly what EfficientNetV2S was pretrained on.

### Workaround (Current v2.0 Deployment)

The Streamlit app sends **raw float32 [0, 255]** pixels to the TFLite model, letting the internal Rescaling layer handle normalization:

```python
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return np.expand_dims(image_array.astype('float32'), axis=0)
```

This gives functional predictions, but the model's accuracy ceiling is limited because it was **trained** on the double-preprocessed (compressed) input distribution while we're now feeding it single-preprocessed input at inference. There is a train/inference distribution mismatch — it works because the internal Rescaling still maps to a range the backbone can interpret, but the learned weights were optimised for a much narrower input band.

### Proper Fix (For v3)

Retrain the model with **one** of these approaches:

**Option A — Remove external `preprocess_input()`** (recommended):
```python
# In _wrap() function, change:
return preprocess_input(img), tf.one_hot(lbl, depth=NUM_CLASSES)
# To:
return img, tf.one_hot(lbl, depth=NUM_CLASSES)
```
Since the model has `include_preprocessing=True`, its internal Rescaling layer will handle `[0, 255] → [-1, 1]` correctly. At deployment, feed raw `[0, 255]` — no external preprocessing needed. This is the cleanest approach.

**Option B — Disable internal preprocessing**:
```python
base_model = EfficientNetV2S(
    weights="imagenet",
    include_top=False,
    include_preprocessing=False,  # ← add this
    input_tensor=inputs
)
```
Keep the external `preprocess_input()` in the pipeline. At deployment, apply `(x / 127.5) - 1.0` before inference. Both paths see `[-1, 1]` — no mismatch.

**Expected impact:** Fixing the double-preprocessing should improve accuracy by **3–5%** (estimated), especially for the currently weak classes that need maximum input discrimination.

---

## 2. Weak Classes (F1 < 0.65)

### Status: **Known Limitation** — mitigated with UI warnings

Three classes consistently underperform:

| Class | Precision | Recall | F1 Score | Training Images | Root Cause |
|---|---|---|---|---|---|
| Priangan Merak Ngibing | 0.40 | 0.40 | 0.40 | 34 | Severely insufficient training data |
| Sogan | 0.43 | 0.38 | 0.40 | ~55 | Visual overlap with other earth-tone Javanese classes |
| Lasem | 0.50 | 0.78 | 0.61 | ~60 | Low precision — model over-predicts this class |

### Current Mitigation

The app displays a warning when:
- Confidence is below 60% (any class)
- The predicted class is one of the three weak classes (regardless of confidence)

### Recommendations for v3

1. **Expand datasets** — minimum 80–100 images per class. Priority: Priangan Merak Ngibing (currently only 34)
2. **Consider Focal Loss** instead of label-smoothed cross-entropy to down-weight easy examples and focus on hard minority classes
3. **Evaluate whether Sogan should remain a separate class** — its earth-tone palette overlaps heavily with Solo Parang and other Javanese court batik styles

---

## 3. TFLite Conversion Workaround

### Status: **Resolved** — workaround is stable

### Problem

When converting a Keras model trained with `mixed_float16` precision to TFLite format, the `TFLiteConverter` enters an infinite loop. This manifests as an endless stream of `TensorSpec(shape=(), dtype=tf.resource, name=None)` log messages with 0% CPU utilization.

### Cause

Keras 3 (shipped with TF 2.16+) injects dynamic `Cast` operations into the computation graph to shuttle values between float16 and float32. The TFLite converter's graph tracing engine cannot resolve these resource-typed variable references and loops indefinitely.

### Workaround (cell7)

Rebuild the model graph in pure float32, then load the trained weights:

```python
tf.keras.mixed_precision.set_global_policy("float32")

inputs = tf.keras.Input(shape=(224, 224, 3))
base = EfficientNetV2S(weights=None, include_top=False, input_tensor=inputs)
# ... rebuild classification head identically ...
model_fp32.load_weights(MODEL_PATH)

with tf.device("/CPU:0"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
```

The clean float32 graph has no `Cast` ops, so conversion completes successfully. FP16 quantization is applied during conversion (not during graph rebuild) for a 2× size reduction.

---

## 4. Recommendations for v3

### Training Pipeline

| Priority | Recommendation | Expected Impact |
|---|---|---|
| **P0** | Fix double-preprocessing bug (see §1) | +3–5% accuracy |
| **P0** | Expand Priangan Merak Ngibing to 80+ images | F1 0.40 → 0.70+ |
| **P1** | Expand Sogan and Lasem to 80+ images each | F1 improvement for both |
| **P1** | Add perceptual hash dedup check between train/test splits | Ensures no data leakage |
| **P2** | Try Focal Loss for class imbalance | Better minority class performance |
| **P2** | Add 1–2 epoch LR warmup in Phase 2 fine-tuning | Smoother gradient transition |
| **P3** | Compute class weights from full dataset, not per-fold | More stable training |

### Deployment

| Priority | Recommendation | Expected Impact |
|---|---|---|
| **P1** | Set up Git LFS for the 39 MB TFLite model | Model actually tracked in repo |
| **P1** | Add basic smoke tests (label consistency, input shape, known-image regression) | Catch bugs before deploy |
| **P2** | Try dynamic range quantization (INT8 weights) | 4× smaller model, faster cold starts |
| **P3** | Add prediction logging for model monitoring | Data for future retraining |

### UI/UX

| Priority | Recommendation | Expected Impact |
|---|---|---|
| **P1** | Color-coded confidence indicator (green/yellow/red) | Better user trust calibration |
| **P1** | Sample gallery showing one typical image per class | Users learn what classes look like |
| **P2** | Prediction history in session state | Compare results across uploads |
| **P2** | Google Fonts + subtle animations | Premium feel |
| **P3** | English/Indonesian language toggle | Broader audience |

---

## Appendix: Diagnostic Script

The following script was used to diagnose the preprocessing bug. Run it against any TFLite model to inspect input expectations:

```python
import tensorflow as tf
import numpy as np

MODEL_PATH = "models/batik_model_v2.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print(f"Input:  shape={inp['shape']} dtype={inp['dtype']}")
print(f"Output: shape={out['shape']} dtype={out['dtype']}")

# Test different input ranges
for label, data in [
    ("zeros",       np.zeros((1,224,224,3), dtype=np.float32)),
    ("raw [0-255]", np.random.rand(1,224,224,3).astype(np.float32) * 255),
    ("scaled [-1,1]", (np.random.rand(1,224,224,3).astype(np.float32) * 2) - 1),
]:
    interpreter.set_tensor(inp['index'], data)
    interpreter.invoke()
    preds = interpreter.get_tensor(out['index'])[0]
    print(f"  {label:20s} -> max={preds.max():.4f} std={preds.std():.4f}")

# Check for internal preprocessing layers
for d in interpreter.get_tensor_details():
    if any(kw in d['name'].lower() for kw in ['rescal', 'preprocess', 'scale']):
        print(f"  Found: {d['name']}")
```

---

*Last updated: 2026-04-29*
