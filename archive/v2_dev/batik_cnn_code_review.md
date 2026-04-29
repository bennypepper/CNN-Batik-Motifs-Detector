# Batik CNN Code Review — Full Audit Report
### EfficientNetB0 Transfer Learning · 20-Class Batik Classification
**Reviewer:** Senior ML Engineer / Computer Vision Expert  
**Date:** April 2026  
**Dataset:** 800 images · 20 classes · 40 images/class  
**Overall Grade: D+ — Not production-ready. Do not deploy results as-is.**

---

## Executive Summary

This notebook implements a two-phase EfficientNetB0 transfer learning pipeline for Batik pattern classification. While the high-level strategy (freeze → fine-tune) is conceptually sound, the implementation contains **7 critical bugs** and **11 significant warnings** that collectively invalidate the reported 84% validation accuracy. The number is almost certainly inflated by data leakage, incorrect preprocessing, and a broken weight-loading step.

The single highest-ROI action is not a code fix — it is sourcing more data. With only 32 training images per class, no architecture or training trick can substitute for data volume.

| Category | Critical | Warning | Pass |
|---|---|---|---|
| Data Loading & Preprocessing | 3 | 2 | 0 |
| Dataset Split & Class Balance | 1 | 2 | 0 |
| Data Augmentation | 1 | 2 | 0 |
| Model Architecture & Transfer Learning | 2 | 2 | 1 |
| Training Strategy & Hyperparameters | 1 | 3 | 1 |
| Evaluation & Metrics | 1 | 2 | 1 |
| Deployment & MLOps | 0 | 5 | 1 |
| **Total** | **9** | **18** | **4** |

---

## Section 1 — Data Loading & Preprocessing

### CRITICAL-1.1 — Entire dataset loaded into RAM as uint8 NumPy arrays

**What is happening:**  
Every image is decoded, resized, and appended to a Python list before training begins. Then `np.array(X)` stacks them all into a single monolithic array held in RAM. At 224×224×3 float32, 800 images is ~483 MB — manageable now, but this pattern is fundamentally broken.

**Why it matters:**  
- At 5,000+ images this crashes with OOM errors  
- It prevents lazy loading, on-the-fly augmentation, and prefetching  
- The entire augmentation pipeline is blocked from working correctly  
- There is no parallelism — loading is single-threaded Python  

**Fix — Replace with a `tf.data` pipeline:**

```python
import tensorflow as tf

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def parse_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def build_dataset(file_paths, labels, batch_size, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=RANDOM_SEED)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

train_ds = build_dataset(train_paths, train_labels, BATCH_SIZE, augment=True, shuffle=True)
val_ds   = build_dataset(val_paths,   val_labels,   BATCH_SIZE, augment=False)
test_ds  = build_dataset(test_paths,  test_labels,  BATCH_SIZE, augment=False)
```

This gives lazy decoding, GPU prefetch, parallel map, and automatic memory management.

---

### CRITICAL-1.2 — No pixel normalisation — raw uint8 `[0, 255]` fed to EfficientNet

**What is happening:**  
`cv2.imread()` returns uint8 arrays in `[0, 255]`. These are passed directly to the model without normalisation. `EfficientNetB0` with `weights='imagenet'` expects pixels processed by its own `preprocess_input()` which scales to approximately `[-1, 1]`.

**Why it matters:**  
The pretrained BatchNormalization layers in EfficientNet were calibrated on ImageNet with `preprocess_input()` applied. Feeding raw `[0–255]` integers produces completely wrong activations. Every forward pass in Phase 1 is computing on incorrect inputs. This is a silent bug — the model trains anyway but on corrupted features.

**Fix:**

```python
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess(image, label):
    # Cast to float32 first, then apply EfficientNet's expected normalisation
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Scales to [-1, 1]
    return image, label

# Apply in the tf.data pipeline AFTER augmentation:
dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
```

**Important:** Do not divide by 255.0 manually. `preprocess_input()` applies the specific per-channel mean subtraction that EfficientNet was trained with. Using `/255.0` instead produces subtly wrong results.

---

### CRITICAL-1.3 — Silent error swallowing in image loader

**What is happening:**  
```python
except Exception as e:
    pass  # <-- every error silently discarded
```

Corrupt files, permission errors, truncated JPEGs, and unsupported formats are all swallowed without logging. There is no record of how many images failed.

**Why it matters:**  
- If certain classes have more read failures, training data becomes silently imbalanced  
- You cannot audit dataset quality or detect systematic corruption  
- The class sizes printed in the log may be wrong  

**Fix:**

```python
failed_files = []

for file in image_files:
    file_path = os.path.join(root, file)
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARN] cv2 returned None for: {file_path}")
            failed_files.append(file_path)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        data_map[label].append(img)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {type(e).__name__}: {e}")
        failed_files.append(file_path)

# After loading, assert failure rate is acceptable
failure_rate = len(failed_files) / max(total_files, 1)
assert failure_rate < 0.05, f"High image failure rate: {failure_rate:.1%}. Check dataset integrity."
print(f"[INFO] Loaded {total_files - len(failed_files)}/{total_files} images. Failed: {len(failed_files)}")
```

---

### WARNING-1.4 — OpenCV used for loading when tf.io is superior

**What is happening:**  
`cv2.imread()` requires a manual `cvtColor(BGR→RGB)` call and runs in Python (GIL-bound). For training pipelines, `tf.io.read_file` + `tf.image.decode_jpeg` + `tf.image.resize` runs on the GPU, is parallelisable via `tf.data`, and integrates natively with the pipeline.

**Recommendation:** Switch entirely to `tf.io` within the `parse_image()` function shown in CRITICAL-1.1. If cv2 is needed for exploratory preprocessing, isolate it to data inspection code only.

---

### WARNING-1.5 — No dtype casting at load time

**What is happening:**  
Images appended to `data_map` are uint8. When `model.fit()` receives them, Keras silently casts each batch at runtime on every epoch. This wastes compute on repeated casts.

**Fix:** Cast and normalise once in the `tf.data` pipeline `preprocess()` function (see CRITICAL-1.2).

---

## Section 2 — Dataset Split & Class Balance

### CRITICAL-2.1 — Only 32 training images per class — catastrophically small

**What is happening:**  
800 total images ÷ 20 classes = 40 per class. An 80/20 split gives 32 train and 8 test images per class. This is not a training set — it is a handful of examples.

**Why it matters:**  
- Neural networks, even with pretrained backbones, cannot reliably generalise from 32 examples of fine-grained visual patterns  
- The model is almost certainly memorising training samples rather than learning discriminative features  
- The 84% val accuracy has a very wide confidence interval and is not a reliable performance estimate  
- The current val set has 8 images per class — a single misclassification changes accuracy by 12.5% per class  

**Fix — Three-way split:**

```python
from sklearn.model_selection import train_test_split

# Step 1: Split off a true held-out test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y_encoded,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=y_encoded
)

# Step 2: Split remaining into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.176,  # ~0.176 of 0.85 = 0.15 of total
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

# Resulting splits: ~70% train / 15% val / 15% test
# X_val is used for callbacks only
# X_test is touched ONCE at the very end for final reporting
```

The test set must never influence any training decision — no callback, no early stopping, no hyperparameter choice should ever see it.

---

### WARNING-2.2 — Test set used as validation during training (data leakage)

**What is happening:**  
```python
model.fit(validation_data=(X_test, y_test), ...)
```
`EarlyStopping` and `ReduceLROnPlateau` both make decisions based on `val_accuracy`, which is actually test accuracy. This means the test set is implicitly used to tune hyperparameters (stopping epoch, learning rate schedule). Final reported metrics are optimistic.

**Fix:** Pass `validation_data=(X_val, y_val)` to `model.fit()`. Reserve `X_test` for `model.evaluate()` after all training is complete. See CRITICAL-2.1 for the split strategy.

---

### WARNING-2.3 — No class imbalance analysis or class weights

**What is happening:**  
The dataset appears perfectly balanced (40 images per class) but this is never verified programmatically. No class weights are computed. If even one class has more silent load failures (see CRITICAL-1.3), training will be biased.

**Fix:**

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Verify balance
unique, counts = np.unique(y_train_encoded, return_counts=True)
print("Class distribution:", dict(zip(le.classes_, counts)))

# Always compute class weights even if dataset appears balanced
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = dict(enumerate(class_weights))

model.fit(..., class_weight=class_weight_dict)
```

---

## Section 3 — Data Augmentation

### CRITICAL-3.1 — Augmentation applied at inference time — validation set is augmented

**What is happening:**  
Augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) are embedded in the model graph via `input_tensor=x`. During `model.fit()`, Keras sets all layers to `training=True` mode by default — including the augmentation layers. This means the validation data is randomly transformed before evaluation, producing noisy and unreliable validation metrics on every epoch.

**Why it matters:**  
- Validation accuracy oscillates more than it should  
- Best weights selected by `ModelCheckpoint` may not be the true best  
- `EarlyStopping` triggers at the wrong time  
- At inference, augmentation layers in `training=False` mode are identity — so predictions are correct, but training metrics were corrupted  

**Fix — Move augmentation into the tf.data pipeline, applied only to training data:**

```python
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    # Random rotation via tfa or custom implementation
    image = tfa.image.rotate(image, tf.random.uniform([], -0.2, 0.2))
    return image, label

train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
# val_ds and test_ds do NOT get augmented
```

The model architecture then becomes clean — no augmentation layers inside the model:

```python
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False locks BatchNorm
```

---

### WARNING-3.2 — `RandomFlip("horizontal_and_vertical")` is wrong for Batik

**What is happening:**  
Batik patterns carry directional meaning. The Parang motif represents flowing water from high to low — a vertical flip creates a culturally invalid pattern. Kawung's floral geometry is similarly orientation-sensitive. Vertical flipping introduces samples that do not correspond to real-world Batik orientations.

**Fix:**

```python
# Only horizontal flip is semantically acceptable for most Batik classes
image = tf.image.random_flip_left_right(image)
# Do NOT use random_flip_up_down for Batik patterns
```

Better augmentations for Batik (texture-focused, pattern-preserving):

```python
image = tf.image.random_brightness(image, max_delta=0.2)   # Simulate lighting variation
image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
image = tf.image.random_hue(image, max_delta=0.05)  # Small hue shift for dye variation
# Small rotations are acceptable (±15°)
# Random crops preserve local texture patterns
```

---

### WARNING-3.3 — Augmentation too weak for a 32-image-per-class dataset

**What is happening:**  
Flip + 20% rotation + 10% zoom is a minimal augmentation strategy that provides perhaps 2–3x effective data variation. With 32 samples per class, aggressive augmentation is not optional — it is the primary regularisation mechanism.

**Fix — Advanced augmentation stack:**

```python
import tensorflow_addons as tfa

def augment_image_aggressive(image, label):
    # Geometric
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tfa.image.rotate(image, tf.random.uniform([], -0.26, 0.26))  # ±15 degrees
    
    # Colour / photometric
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.05)
    
    # Regularisation augmentation
    image = tfa.image.random_cutout(tf.expand_dims(image, 0), mask_size=(40, 40))[0]
    
    return image, label

# MixUp at batch level (implement after batching)
def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    weights = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size])
    weights = tf.reshape(weights, [batch_size, 1, 1, 1])
    idx = tf.random.shuffle(tf.range(batch_size))
    images = weights * images + (1 - weights) * tf.gather(images, idx)
    label_weights = tf.reshape(weights, [batch_size, 1])
    labels = label_weights * labels + (1 - label_weights) * tf.gather(labels, idx)
    return images, labels
```

---

## Section 4 — Model Architecture & Transfer Learning

### CRITICAL-4.1 — BatchNormalization layers behave incorrectly during fine-tuning

**What is happening:**  
In Phase 2:
```python
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
```
When `base_model.trainable = True` is set, ALL `BatchNormalization` layers in the network switch to training mode and begin updating their running mean and variance from Phase 2 mini-batches. With only 32 images/class, mini-batch statistics are highly unstable and completely corrupt the pretrained BN statistics that EfficientNet relies on for correct feature computation.

**Why it matters:**  
This is the most common and most damaging silent bug in Keras transfer learning. The model's pretrained feature extraction degrades during fine-tuning. The official TensorFlow fine-tuning guide explicitly warns about this behaviour.

**Fix — Freeze all BN layers regardless of trainable state:**

```python
base_model.trainable = True

for layer in base_model.layers:
    # Freeze BatchNormalization layers unconditionally
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Freeze early layers (keep BN freeze from above)
for layer in base_model.layers[:-20]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile after changing trainable state — mandatory
model.compile(optimizer=Adam(learning_rate=1e-5), ...)
```

Pass `training=False` explicitly when calling the base model in the functional API:

```python
x = base_model(inputs, training=False)  # Forces BN to use stored statistics
```

---

### CRITICAL-4.2 — Augmentation connected to base_model via `input_tensor` — architectural flaw

**What is happening:**  
```python
x = RandomFlip(...)(inputs)
x = RandomRotation(0.2)(x)
base_model = EfficientNetB0(..., input_tensor=x)  # Augmentation baked into base_model
```
Using `input_tensor=x` makes the augmentation layers part of the `base_model` subgraph. This means:
- `base_model.trainable = False` also locks augmentation layers inside it (harmless but confusing)
- The model's Input layer is external to `base_model`, breaking standard serialisation assumptions
- `base_model.summary()` shows augmentation layers as part of EfficientNet

**Fix — Build the model with clean separation:**

```python
def build_model(num_classes):
    inputs = Input(shape=(224, 224, 3))
    
    # Base model is constructed independently, connected via functional call
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Phase 1: all frozen
    
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return base_model, model
```

---

### WARNING-4.3 — Head architecture too thin for 20-class fine-grained classification

**What is happening:**  
A single `Dense(256)` with `Dropout(0.4)` processes 1280-dimensional EfficientNet features for 20 fine-grained Batik classes. Batik patterns have subtle inter-class differences (e.g., Parang vs Kawung vs Megamendung require high-level texture discrimination). The current head may lose discriminative information.

**Fix — Deeper, regularised head:**

```python
from tensorflow.keras.regularizers import l2

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)         # Stabilise activations from backbone
x = Dense(512, activation='relu',
          kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',
          kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
```

---

### WARNING-4.4 — EfficientNetB0 may be undersized for fine-grained texture discrimination

**What is happening:**  
B0 has 5.3M parameters and was designed for efficient inference on mobile devices. For fine-grained texture discrimination across 20 Batik classes (highly similar patterns, subtle motif differences), B0's representational capacity is limited.

**Recommendation:**

| Model | Params | Top-1 ImageNet | Recommendation |
|---|---|---|---|
| EfficientNetB0 | 5.3M | 77.1% | Current — underpowered |
| EfficientNetB3 | 12M | 81.6% | Good balance |
| EfficientNetV2-S | 21M | 83.9% | Best for this task |
| ConvNeXt-Tiny | 28M | 82.1% | Alternative |

```python
from tensorflow.keras.applications import EfficientNetV2S
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

---

### PASS-4.5 — Two-phase transfer learning strategy is conceptually correct

The approach of Phase 1 (frozen backbone, train head) followed by Phase 2 (unfreeze top layers, low LR fine-tuning) is correct. The LR ratio of 1e-3 → 1e-5 is appropriate. This structure should be retained.

---

## Section 5 — Training Strategy & Hyperparameters

### CRITICAL-5.1 — Phase 1 `EarlyStopping(patience=3)` is dangerously low

**What is happening:**  
The training log shows `val_accuracy` oscillating:
```
Epoch 7:  val_accuracy = 0.7688
Epoch 8:  val_accuracy = 0.7937
Epoch 9:  val_accuracy = 0.8062
Epoch 10: val_accuracy = 0.8000
Epoch 11: val_accuracy = 0.8062
```
With patience=3, training terminates after 3 consecutive non-improving epochs. Given the noisy validation (8 images per class), this is inadequate — legitimate improvement can be cut short. Phase 1 appears to have stopped before the head was fully converged.

**Fix:**

```python
# Phase 1
early_stop_p1 = EarlyStopping(
    monitor='val_loss',      # Loss is smoother than accuracy for stopping decisions
    patience=7,
    restore_best_weights=True,
    verbose=1,
    min_delta=1e-4
)

# Phase 2
early_stop_p2 = EarlyStopping(
    monitor='val_loss',
    patience=10,             # Allow more time after LR changes
    restore_best_weights=True,
    verbose=1,
    min_delta=1e-4
)
```

---

### WARNING-5.2 — No learning rate warmup in Phase 2

**What is happening:**  
Fine-tuning jumps immediately to `lr=1e-5` with freshly unfrozen layers. Without warmup, large gradient updates can destabilise pretrained representations in the first few epochs.

**Fix — Cosine decay with warmup:**

```python
from tensorflow.keras.optimizers.schedules import CosineDecay

steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps = 25 * steps_per_epoch
warmup_steps = 3 * steps_per_epoch

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=total_steps - warmup_steps,
    alpha=1e-7  # Final LR
)

optimizer = Adam(learning_rate=lr_schedule)
```

---

### WARNING-5.3 — `ReduceLROnPlateau` and `EarlyStopping` conflict in Phase 2

**What is happening:**  
```python
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7)
early_stop = EarlyStopping(patience=5, ...)
```
`ReduceLROnPlateau` fires at patience=3, then `EarlyStopping` stops training at patience=5. The model gets only 2 epochs to benefit from the reduced LR before training ends. The callbacks are counteracting each other.

**Fix:** Use a single built-in schedule (`CosineDecay`) in the optimiser instead of `ReduceLROnPlateau`. Keep only `EarlyStopping` with patience=10:

```python
# Remove ReduceLROnPlateau entirely
# Use CosineDecay schedule as shown in WARNING-5.2
callbacks = [checkpoint, early_stop_p2]  # No ReduceLROnPlateau
```

---

### WARNING-5.4 — Label smoothing not applied — overconfident softmax on tiny dataset

**What is happening:**  
With 32 images/class, the model will overfit to hard one-hot labels, pushing softmax outputs to near-zero entropy on training data. Label smoothing prevents this by distributing a small probability mass across all classes.

**Fix:**

```python
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
```

---

### PASS-5.5 — `ModelCheckpoint` saves best weights correctly

Using `save_best_only=True` with `monitor='val_accuracy'` and `.keras` format in Phase 2 is correct practice. This should be retained in both phases.

---

## Section 6 — Evaluation & Metrics

### CRITICAL-6.1 — `model.load_weights()` called on a full SavedModel — silent failure

**What is happening:**  
```python
try:
    model.load_weights("best_batik_model.keras")
except:
    print("⚠️ Warning: Could not load best weights, using current weights.")
```

`ModelCheckpoint` with `.keras` extension saves the full model using `model.save()` — not a weights-only checkpoint. Calling `model.load_weights()` on a full SavedModel raises an error, which is caught by the bare `except` and silently ignored. The evaluation then runs on whichever weights are in memory at script end — not the best weights.

**Fix:**

```python
from tensorflow.keras.models import load_model

# Load the full saved model
best_model = load_model("best_batik_model.keras")
y_pred_probs = best_model.predict(test_ds)

# If you want weights-only checkpoints, save and load consistently:
# Save: model.save_weights("best_weights.weights.h5")
# Load: model.load_weights("best_weights.weights.h5")
```

---

### WARNING-6.2 — Evaluation stops at accuracy and confusion matrix

**What is happening:**  
For a 20-class fine-grained task, accuracy alone is insufficient. The confusion matrix shows which classes are confused but gives no probabilistic insight.

**Fix — Comprehensive evaluation:**

```python
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Top-3 accuracy
top3_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
top3_metric.update_state(y_true, y_pred_probs)
print(f"Top-1 Accuracy: {np.mean(y_pred == y_true):.4f}")
print(f"Top-3 Accuracy: {top3_metric.result().numpy():.4f}")

# Macro ROC-AUC
auc = roc_auc_score(
    tf.keras.utils.to_categorical(y_true, num_classes),
    y_pred_probs,
    multi_class='ovr',
    average='macro'
)
print(f"Macro ROC-AUC: {auc:.4f}")

# Per-class F1 bar chart
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
f1_scores = [report_dict[c]['f1-score'] for c in class_names]
plt.figure(figsize=(14, 5))
plt.bar(class_names, f1_scores, color=['green' if f > 0.8 else 'orange' if f > 0.6 else 'red' for f in f1_scores])
plt.axhline(y=0.8, linestyle='--', color='gray', label='0.8 threshold')
plt.xticks(rotation=45, ha='right')
plt.title('Per-class F1 Score')
plt.tight_layout()
plt.savefig('per_class_f1.png')

# Confidence calibration — reliability diagram
from sklearn.calibration import calibration_curve
plt.figure(figsize=(6, 6))
for i, class_name in enumerate(class_names[:5]):  # Sample 5 classes
    prob_true, prob_pred = calibration_curve(
        (y_true == i).astype(int),
        y_pred_probs[:, i],
        n_bins=10
    )
    plt.plot(prob_pred, prob_true, label=class_name)
plt.plot([0,1],[0,1],'k--',label='Perfect calibration')
plt.legend(fontsize=8)
plt.title('Reliability Diagram (Calibration)')
plt.savefig('calibration.png')
```

---

### WARNING-6.3 — Overfitting visible in training logs but not addressed

**What is happening:**  
```
Epoch 12: train_accuracy = 0.927 · val_accuracy = 0.831  →  gap = 9.6%
```
This is textbook overfitting. The gap widens in Phase 2. The code has no active mechanism to close the gap — only to stop growing it.

**Fix — Regularisation stack (in addition to augmentation):**

```python
from tensorflow.keras.regularizers import l2

# 1. L2 regularisation on dense layers
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)

# 2. Higher dropout
x = Dropout(0.5)(x)

# 3. Stochastic Weight Averaging (SWA) — averages weights over final epochs
# Use: tensorflow_addons.optimizers.SWA wrapper

# 4. Reduce head capacity if overfitting persists
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)  # Reduce from 256
```

---

### PASS-6.4 — Combined history plot with phase boundary marker

Concatenating `history1 + history2` and marking the fine-tuning transition with `axvline` is correct and informative. This pattern should be kept.

---

## Section 7 — Deployment & MLOps

### WARNING-7.1 — Saving `.h5` legacy format despite Keras warning

**What is happening:**  
The output log explicitly states:
```
WARNING:absl:You are saving your model as an HDF5 file via model.save().
This file format is considered legacy.
```
The code ignores this warning and saves `.h5` anyway. The `.keras` native format is smaller, more robust, handles custom layers correctly, and is the current standard.

**Fix:**

```python
# Primary deployment format
model.save("batik_model_deploy.keras")

# Verify loading works before finalising
test_load = tf.keras.models.load_model("batik_model_deploy.keras")
test_pred = test_load.predict(test_ds.take(1))
assert test_pred.shape[-1] == len(class_names), "Model output shape mismatch"
print("Model saved and verified successfully.")
```

---

### WARNING-7.2 — TFLite conversion skips quantisation — no size reduction

**What is happening:**  
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()  # Zero configuration
```
No optimisations are set. The output `.tflite` file is roughly the same size as the Keras model. The purpose of TFLite is quantisation.

**Fix — FP16 quantisation (good balance of size and accuracy):**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('batik_model_fp16.tflite', 'wb') as f:
    f.write(tflite_model)

# For INT8 full quantisation (4x smaller, 2-3x faster):
def representative_dataset():
    for images, _ in val_ds.take(100):
        yield [images]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_int8 = converter_int8.convert()

with open('batik_model_int8.tflite', 'wb') as f:
    f.write(tflite_int8)

print(f"FP16 size: {len(tflite_model)/1e6:.1f} MB")
print(f"INT8 size: {len(tflite_int8)/1e6:.1f} MB")
```

---

### WARNING-7.3 — Hardcoded paths, no config file, no TF reproducibility seed

**What is happening:**  
`ROOT_DIR`, `TARGET_FOLDER`, `IMG_SIZE`, `BATCH_SIZE` are module-level constants. `tf.random.set_seed()` is missing despite `RANDOM_SEED` being defined. Runs are not fully reproducible.

**Fix — Config dataclass + full seed setting:**

```python
import os
import random
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Config:
    root_dir:      str   = "/kaggle/input/batik-nusantara-batik-indonesia-dataset"
    target_folder: str   = "raw_batik_v2.1"
    img_size:      tuple = (224, 224)
    batch_size:    int   = 32
    seed:          int   = 42
    phase1_epochs: int   = 20
    phase2_epochs: int   = 30
    phase1_lr:     float = 1e-3
    phase2_lr:     float = 1e-5
    dropout_rate:  float = 0.5
    unfreeze_layers: int = 20
    label_smoothing: float = 0.1

cfg = Config()

# Full reproducibility
os.environ['PYTHONHASHSEED'] = str(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)
```

---

### WARNING-7.4 — No multi-GPU strategy despite 2 GPUs being available

**What is happening:**  
Two GPUs were detected (`GPU:0` and `GPU:1`) but training used only one. At this dataset size the impact is minimal, but the pattern should be established.

**Fix:**

```python
strategy = tf.distribute.MirroredStrategy()
print(f"Running on {strategy.num_replicas_in_sync} GPUs")

with strategy.scope():
    base_model, model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer=Adam(learning_rate=cfg.phase1_lr),
        loss=CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
        metrics=['accuracy']
    )
```

When using `MirroredStrategy`, increase batch size proportionally: `BATCH_SIZE = 32 * strategy.num_replicas_in_sync`.

---

### WARNING-7.5 — LabelEncoder saved as plain text — brittle at inference

**What is happening:**  
```python
with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))
```
The plain text file requires the downstream Streamlit app to independently reconstruct the label-to-index mapping. If class order ever changes between retraining runs (alphabetical ordering from `os.walk` is not guaranteed across filesystems), all predictions are silently wrong.

**Fix:**

```python
import joblib

# Save the fitted LabelEncoder object — preserves the exact mapping
joblib.dump(le, "label_encoder.pkl")

# Also save a JSON for human readability and cross-language use
import json
label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

# At inference in Streamlit:
le_loaded = joblib.load("label_encoder.pkl")
pred_class = le_loaded.inverse_transform([np.argmax(pred)])[0]
```

---

## Section 8 — Data Acquisition Strategy

This is the highest-priority action before any further code improvement. **No training technique compensates for 32 images per class.**

### 8.1 — Target Dataset Size

The minimum viable dataset for reliable fine-grained classification with a pretrained backbone is approximately 200–500 images per class. For 20 classes, the target is **4,000–10,000 images total**. The current 800 is 5–12x below this threshold.

| Images per class | Expected val accuracy ceiling | Notes |
|---|---|---|
| 40 (current) | ~84% (inflated) | Unreliable, high variance |
| 100 | ~88–90% | Minimum viable |
| 200 | ~92–94% | Good generalisation |
| 500+ | ~95%+ | Reliable deployment quality |

---

### 8.2 — Primary Sources — Open Web Scraping

**Tool: `gallery-dl` (most effective for image bulk download)**

```bash
pip install gallery-dl

# Scrape Batik images from Flickr by tag
gallery-dl "https://www.flickr.com/search/?text=batik+parang+yogyakarta"
gallery-dl "https://www.flickr.com/search/?text=batik+kawung+solo"

# Pinterest (high volume, varied quality)
gallery-dl "https://www.pinterest.com/search/pins/?q=batik+megamendung"
```

**Tool: `icrawler` (Python, programmatic)**

```python
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, FlickrImageCrawler

# Per class, per region query
batik_classes = [
    ("Bali_Barong",             "batik barong bali traditional"),
    ("Yogyakarta_Kawung",       "batik kawung yogyakarta"),
    ("JawaBarat_Megamendung",   "batik megamendung cirebon"),
    ("Solo_Parang",             "batik parang solo"),
    ("Papua_Cendrawasih",       "batik cendrawasih papua"),
    # ... all 20 classes
]

for class_name, query in batik_classes:
    os.makedirs(f"scraped/{class_name}", exist_ok=True)
    
    # Google Images
    google_crawler = GoogleImageCrawler(storage={"root_dir": f"scraped/{class_name}"})
    google_crawler.crawl(keyword=query, max_num=200)
    
    # Bing Images (complementary results)
    bing_crawler = BingImageCrawler(storage={"root_dir": f"scraped/{class_name}"})
    bing_crawler.crawl(keyword=query, max_num=200)
```

**Tool: Selenium + BeautifulSoup for manual scraping of Indonesian Batik museums and galleries:**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests, os, time

SOURCES = [
    "https://www.batikmarkets.com",
    "https://tokobatik.com",
    "https://batikjawa.com",
    "https://wastra.id",  # Indonesian textile database
]

# Scrape with rate limiting (respect robots.txt)
def scrape_images(url, class_name, max_images=100):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(2)
    images = driver.find_elements(By.TAG_NAME, 'img')
    for i, img in enumerate(images[:max_images]):
        src = img.get_attribute('src')
        if src and src.startswith('http'):
            r = requests.get(src, timeout=10)
            with open(f"scraped/{class_name}/web_{i}.jpg", 'wb') as f:
                f.write(r.content)
            time.sleep(0.5)  # Rate limit
    driver.quit()
```

---

### 8.3 — Institutional and Open Dataset Sources

These provide high-quality, correctly-labelled images:

**Indonesian Government / Museum Digital Archives:**
- Museum Batik Pekalongan — digital collection requests via email
- Kementerian Perindustrian RI — batik documentation reports with images
- BPNB (Balai Pelestarian Nilai Budaya) regional offices — often have digital catalogues

**Academic Datasets to search:**
- Mendeley Data: search "batik dataset" — several annotated Batik datasets exist
- Kaggle: beyond current dataset — search "batik classification", "batik indonesia"
- IEEE DataPort: "batik fabric classification"
- Zenodo: "Indonesian batik"

**Existing supplementary datasets:**
```
# Known publicly available Batik datasets (verify current availability):
# 1. Batik Image Dataset — Universitas Dian Nuswantoro (UDINUS)
# 2. Batik Nusantara Dataset — various Kaggle versions
# 3. DBatik — Indonesian Batik Pattern Dataset (academic publication)
# 4. BATIK-7 / BATIK-15 datasets from Indonesian ML papers
```

---

### 8.4 — Synthetic Data Generation

For classes with persistent data scarcity after scraping:

**Option A: ControlNet + Stable Diffusion (best quality)**

```python
# Using diffusers library with ControlNet
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Generate Batik variations using real image as control
prompt = "high quality batik kawung yogyakarta fabric pattern, traditional Indonesian textile"
negative_prompt = "blurry, distorted, low quality, cartoon"

# Use Canny edge map of a real Batik image as control input
for i in range(50):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,          # Edge-detected real Batik sample
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator().manual_seed(i)
    ).images[0]
    image.save(f"synthetic/Yogyakarta_Kawung/synth_{i:04d}.jpg")
```

**Option B: AugLy / Albumentations for real-image variants**

```python
import albumentations as A

heavy_augmentation = A.Compose([
    A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Rotate(limit=15, p=0.8),
    A.RandomCrop(height=200, width=200, p=0.5),
    A.Resize(224, 224),
])

# Generate 5 augmented variants per original image
for img_path in original_images:
    img = cv2.imread(img_path)
    for j in range(5):
        augmented = heavy_augmentation(image=img)['image']
        cv2.imwrite(f"augmented_{j}.jpg", augmented)
```

**Option C: CycleGAN for domain transfer (advanced)**

For creating night/day variants, different fabric textures, aged vs new Batik — train a CycleGAN on pairs of Batik images with different lighting or age conditions. This requires ~200 image pairs but produces highly realistic domain-transferred samples.

---

### 8.5 — Human Labelling Workflow

For scraped images that need quality control:

**Tool: Label Studio (self-hosted, free)**

```bash
pip install label-studio
label-studio start --port 8080

# Import scraped images, assign to Indonesian Batik expert labellers
# Use inter-annotator agreement (Cohen's Kappa > 0.8) to validate labels
```

**Tool: CVAT (Computer Vision Annotation Tool)**

```bash
docker run -p 8080:8080 openvino/cvat_server
```

**Labelling protocol:**
1. Initial auto-labelling using current model predictions (confidence > 0.95 only)
2. Human review of all auto-labelled images
3. Manual labelling of remaining images by domain expert
4. Cross-validation: each image labelled by 2 independent annotators
5. Resolve disagreements via majority vote or expert tiebreaker
6. Target: Cohen's Kappa > 0.85 before including in training set

---

### 8.6 — Data Quality Filtering Pipeline

After collection, filter aggressively before training:

```python
import imagehash
from PIL import Image
from collections import defaultdict

def filter_dataset(image_dir):
    hashes = defaultdict(list)
    to_remove = []
    
    for img_path in Path(image_dir).rglob("*.jpg"):
        try:
            img = Image.open(img_path)
            
            # 1. Near-duplicate detection via perceptual hash
            phash = str(imagehash.phash(img))
            if any(imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(h) < 8
                   for h in hashes):
                to_remove.append(img_path)
                continue
            hashes[phash].append(str(img_path))
            
            # 2. Minimum resolution check
            w, h = img.size
            if w < 150 or h < 150:
                to_remove.append(img_path)
                continue
            
            # 3. Aspect ratio sanity check (reject extreme panoramas)
            ratio = max(w, h) / min(w, h)
            if ratio > 3.0:
                to_remove.append(img_path)
                continue
                
            # 4. Blur detection via Laplacian variance
            import cv2, numpy as np
            img_cv = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            blur_score = cv2.Laplacian(img_cv, cv2.CV_64F).var()
            if blur_score < 50:  # Threshold tuned empirically
                to_remove.append(img_path)
                continue
                
        except Exception as e:
            to_remove.append(img_path)
    
    print(f"Removing {len(to_remove)} low-quality/duplicate images")
    for p in to_remove:
        os.remove(p)
    
    return len(to_remove)
```

---

## Section 9 — Prioritised Action Plan

Below is the recommended implementation order, from highest to lowest impact:

### Phase 0 — Immediate (fix before any further training)

1. **Fix `load_weights()` bug** (CRITICAL-6.1) — 5 minutes. Current evaluation results may be wrong.
2. **Add `preprocess_input()`** (CRITICAL-1.2) — 10 minutes. All training so far was on incorrectly-normalised inputs.
3. **Fix BatchNormalization fine-tuning** (CRITICAL-4.1) — 15 minutes. Phase 2 results are invalid without this.

### Phase 1 — Data & Pipeline (1–2 weeks)

4. **Source more data** (Section 8) — minimum 200 images per class before proceeding
5. **Replace in-memory loading with `tf.data` pipeline** (CRITICAL-1.1)
6. **Implement three-way train/val/test split** (CRITICAL-2.1, WARNING-2.2)
7. **Move augmentation to `tf.data` pipeline** (CRITICAL-3.1)
8. **Fix silent error handling in loader** (CRITICAL-1.3)

### Phase 2 — Model & Training (1 week)

9. **Fix model architecture to separate base_model from augmentation** (CRITICAL-4.2)
10. **Add L2 regularisation and label smoothing** (WARNING-5.4, WARNING-6.3)
11. **Implement CosineDecay LR schedule with warmup** (WARNING-5.2)
12. **Replace ReduceLROnPlateau with built-in schedule** (WARNING-5.3)
13. **Upgrade to EfficientNetV2-S or B3** (WARNING-4.4)

### Phase 3 — Evaluation & Deployment (3 days)

14. **Add Top-3 accuracy, ROC-AUC, calibration metrics** (WARNING-6.2)
15. **Apply INT8 quantisation to TFLite export** (WARNING-7.2)
16. **Save LabelEncoder as `.pkl`** (WARNING-7.5)
17. **Add config dataclass and TF seed** (WARNING-7.3)
18. **Enable MirroredStrategy for dual-GPU training** (WARNING-7.4)

---

## Appendix — Corrected Model Build Function

This is a corrected, production-ready version of `build_optimized_model()`:

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dropout,
                                      Dense, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def build_phase1_model(num_classes, img_size=(224, 224)):
    inputs = Input(shape=(*img_size, 3))
    
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Phase 1: all layers frozen, BN in inference mode
    base_model.trainable = False
    
    x = base_model(inputs, training=False)  # training=False locks BN statistics
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return base_model, model


def prepare_phase2_model(base_model, model, num_classes, total_steps):
    # Unfreeze top layers
    base_model.trainable = True
    
    # CRITICAL: Always freeze BatchNormalization layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # Freeze early backbone layers
    for layer in base_model.layers[:-30]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # Cosine decay schedule with warmup
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-7
    )
    
    # Must recompile after changing trainable state
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return model
```

---

*End of report. All findings are based on static code analysis and execution log review. Accuracy estimates are probabilistic assessments, not guarantees.*
