# Architecture: CNN Batik Motifs Detector v2

This document is the authoritative technical reference for the v2 Batik Motif classification pipeline. It covers every design decision from dataset curation through model deployment.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Pipeline Structure](#3-pipeline-structure)
4. [Cell 1 — Environment & Configuration](#4-cell-1--environment--configuration)
5. [Cell 2 — EDA & Integrity Check](#5-cell-2--eda--integrity-check)
6. [Cell 3 — Data Pipeline & Augmentation](#6-cell-3--data-pipeline--augmentation)
7. [Cell 4 — K-Fold Training (Phase 1 + Phase 2)](#7-cell-4--k-fold-training-phase-1--phase-2)
8. [Cell 5 — K-Fold Summary & Final Model](#8-cell-5--k-fold-summary--final-model)
9. [Cell 6 — Evaluation with TTA](#9-cell-6--evaluation-with-tta)
10. [Cell 7 — TFLite Export](#10-cell-7--tflite-export)
11. [Cell 8 — Output Packaging](#11-cell-8--output-packaging)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Project Overview

The goal is to classify Indonesian batik motifs from images into one of 28 distinct classes using a fine-tuned convolutional neural network. The model is deployed as a quantized TFLite artifact inside a Streamlit web application hosted on Streamlit Community Cloud.

**Key constraints:**
- Training environment: Kaggle Notebook, NVIDIA Tesla P100 GPU
- Deployment target: Streamlit Community Cloud (CPU inference)
- Dataset size: 2,216 images across 28 classes (small dataset regime)

---

## 2. Dataset

### Source
Derived from the [Batik-Indonesia dataset](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia) by **Muhammad Salman Al Faridzi**, Apache 2.0 License.

Hosted on Kaggle as: `indonesian-batik-dataset-enhanced-and-cleaned`

### Modifications from Original (38 → 28 Classes)

**10 classes removed** due to poor visual discriminability for CNN training:

| Class | Reason for Removal |
|---|---|
| `Bali` | Geographic umbrella — multiple incompatible visual variants |
| `Batik` | Redundant meta-label |
| `Ciamis` | Extreme intra-class variance |
| `Garutan` | Extreme intra-class variance |
| `Pekalongan` | Geographic umbrella — too many variants |
| `Betawi` | Visual overlap with retained `Jakarta_Ondel_Ondel` class |
| `Keraton` | Generic royal patterns with high ambiguity |
| `Sidomukti` | Overlaps with `Sidoluhur` and other Sido variants |
| `Celup` / `Gentongan` | Dyeing *techniques*, not spatial motifs |
| `Priangan` | Too broad — replaced with specific subtype |
| `Aceh` | Insufficient samples for reliable training |

**1 class replaced:**
- `Priangan` → `Priangan_Merak_Ngibing`: Focused exclusively on the *Merak Ngibing* (Dancing Peacock) subtype, which has a distinct, repeatable geometric pattern the CNN can learn.

**4 classes expanded** (manually sourced and verified images added):
`Sogan`, `Lasem`, `Ceplok`, `Priangan_Merak_Ngibing`

### Final Statistics

| Metric | Value |
|---|---|
| Total images | 2,216 |
| Total classes | 28 |
| Mean per class | 79.1 images |
| Min (`Priangan_Merak_Ngibing`) | 34 images |
| Max (`Kalimantan_Dayak`) | 230 images |
| Class imbalance ratio | ~6.8× |
| Formats accepted | `.jpg`, `.jpeg`, `.png`, `.webp` |

### Augmentation Tier Assignment

Classes are assigned an augmentation tier based on total image count, computed once at Cell 3 runtime from the full `df`:

| Tier | Threshold | Classes | Strategy |
|---|---|---|---|
| Tier 1 | ≤ 45 images | 1 (`Priangan_Merak_Ngibing`) | Aggressive augmentation |
| Tier 2 | 46–90 images | 22 classes | Moderate augmentation |
| Tier 3 | > 90 images | 5 classes | Light augmentation |

---

## 3. Pipeline Structure

The pipeline is split into 8 sequential Kaggle notebook cells. All cells share global state within a single Kaggle session.

```
Cell 1  → Global config, imports, GPU setup, mixed precision
Cell 2  → EDA, integrity checks, class distribution visualization
Cell 3  → Test holdout split, K-Fold setup, augmentation pipelines, tf.data functions
Cell 4  → 5-Fold K-Fold loop (Phase 1 + Phase 2 per fold), saves fold models
Cell 5  → K-Fold summary report, final full-data model training (Phase 1 + Phase 2)
Cell 6  → Evaluation with Test-Time Augmentation (TTA), confusion matrix, F1 chart
Cell 7  → TFLite FP16 export (float32 graph rebuild to avoid Keras 3 deadlock)
Cell 8  → Zip all outputs for single-click download
```

---

## 4. Cell 1 — Environment & Configuration

### Reproducibility
Seeds are fixed across all libraries — Python `random`, NumPy, and TensorFlow — plus environment variables `TF_DETERMINISTIC_OPS` and `TF_CUDNN_DETERMINISTIC` to ensure deterministic CUDA behavior on the P100.

### Mixed Precision
`mixed_float16` is enabled globally via `tf.keras.mixed_precision.set_global_policy`. This halves memory bandwidth usage and nearly doubles throughput on Tensor Core GPUs (P100/T4/A100). The classification head's final activation is explicitly cast to `float32` to maintain numerical stability in the softmax.

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `BATCH_SIZE` | 32 | Balanced for P100 VRAM with EfficientNetV2S |
| `N_FOLDS` | 5 | Full stratified K-Fold coverage |
| `PHASE1_LR` | 1e-3 | Standard Adam LR for frozen backbone training |
| `PHASE2_LR` | 1e-5 | Conservative LR to gently nudge pretrained weights |
| `UNFREEZE_LAYERS` | 40 | Tail layers of EfficientNetV2S unfrozen for fine-tuning |
| `DROPOUT_RATE_1` | 0.5 | After GAP — high regularization for head |
| `DROPOUT_RATE_2` | 0.3 | After Dense — lighter second dropout |
| `DENSE_UNITS` | 256 | Classification head intermediate size |
| `LABEL_SMOOTHING` | 0.1 | Prevents overconfidence on ambiguous motifs |
| `TTA_STEPS` | 5 | TTA inference passes at evaluation time |
| `TEST_SIZE` | 0.15 | Fixed 15% holdout, stratified |
| `TIER1_THRESHOLD` | 45 | Images below this get aggressive augmentation |
| `TIER2_THRESHOLD` | 90 | Images below this get moderate augmentation |

---

## 5. Cell 2 — EDA & Integrity Check

Every image in the dataset is scanned with OpenCV before training:
- **Corruption check:** `cv2.imread` returns `None` for unreadable files — these are dropped.
- **Dimension check:** Images smaller than 32×32 px are dropped (untrainable noise).

Outputs saved to `/kaggle/working`:
- `eda_class_distribution.png` — horizontal bar chart with tier color coding and threshold lines
- `eda_sample_grid.png` — 4 sample images per class in a grid layout

---

## 6. Cell 3 — Data Pipeline & Augmentation

### Data Splitting Strategy

A **fixed 15% stratified test holdout** is carved out first and never touched during training or validation. This is the gold standard evaluation set used in Cell 6.

The remaining 85% (`X_trainval`) is used exclusively for K-Fold cross-validation in Cell 4. There is no separate static validation split — the fold itself provides the validation set per iteration.

```
Full dataset (2,216)
└── Fixed Test Holdout — 15% (stratified, random_state=42)
└── Train+Val Pool — 85%
    ├── Fold 1: Train 80% | Val 20%
    ├── Fold 2: Train 80% | Val 20%
    ├── Fold 3: Train 80% | Val 20%
    ├── Fold 4: Train 80% | Val 20%
    └── Fold 5: Train 80% | Val 20%
```

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ensures class proportions are maintained in every fold split.

### Augmentation Pipelines (Albumentations 1.4.3)

Augmentation is applied **inside `tf.data` via `tf.py_function`**, which means:
- Training images are augmented randomly on-the-fly (no leakage)
- Validation/test images receive only resize — never augmented
- Each epoch sees a different random augmentation for each training image

The tier assigned to a class is fixed from the full dataset image count, not the fold's training subset. This ensures consistent augmentation intensity regardless of which fold is active.

**Tier 1 — Aggressive** (for `Priangan_Merak_Ngibing`, 34 images):
- HorizontalFlip, RandomBrightnessContrast, HueSaturationValue
- GridDistortion, ElasticTransform (simulate fabric warp)
- RandomResizedCrop (simulate varied framing)
- GaussianBlur, CoarseDropout (simulate occlusion/noise)
- CLAHE (local contrast enhancement), Sharpen

**Tier 2 — Moderate** (22 classes, 46–90 images):
- HorizontalFlip, RandomBrightnessContrast, HueSaturationValue
- RandomResizedCrop, GaussianBlur, CLAHE

**Tier 3 — Light** (5 classes, >90 images):
- HorizontalFlip, mild RandomBrightnessContrast, mild HueSaturationValue

**Validation / TTA pipeline:**
- Resize only (no augmentation) for standard evaluation
- Tier 3 light augmentation reused as the TTA pipeline (random but mild)

### tf.data Pipeline

```
from_tensor_slices → shuffle (train only) → map(augment_fn, parallel) → batch → prefetch
```

- `num_parallel_calls=AUTOTUNE` — augmentation runs in parallel across CPU cores
- `prefetch(AUTOTUNE)` — next batch prepared while GPU trains on current batch
- `cache=True` for test/val datasets — images decoded once, cached in RAM
- `reshuffle_each_iteration=True` for training — ensures different order each epoch

**Preprocessing:** After augmentation, `preprocess_input` from `efficientnet_v2` module is applied (scales pixel values to the range EfficientNetV2S expects from its ImageNet pretraining).

---

## 7. Cell 4 — K-Fold Training (Phase 1 + Phase 2)

### Why K-Fold?

With only ~79 images per class on average, a single random split is statistically unreliable. A lucky or unlucky split can swing macro F1 by 5–10%. Stratified K-Fold guarantees that **every image in the dataset is used for validation exactly once**, giving a statistically robust estimate of true model performance.

### Model Architecture — EfficientNetV2S

**Why EfficientNetV2S over B0?**
- EfficientNetV2S has ~20M parameters vs. ~5M for B0 — better feature extraction capacity
- V2 uses Fused-MBConv blocks in early layers — faster training convergence
- V2S achieves higher ImageNet accuracy at comparable or better speed on P100
- The 224×224 input size is fully supported — no resolution change required

**Architecture:**
```
Input (224, 224, 3)
    ↓
EfficientNetV2S backbone (pretrained ImageNet, 400+ layers)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization       ← stabilizes activation distribution from backbone
    ↓
Dropout(0.5)
    ↓
Dense(256, relu)
    ↓
Dropout(0.3)
    ↓
Dense(28)               ← logits, no activation
    ↓
Activation("softmax", dtype="float32")   ← explicit float32 for mixed precision stability
```

### Training Phases (per fold)

**Phase 1 — Frozen Backbone, Head Training:**
- Entire EfficientNetV2S backbone is frozen (`trainable=False`)
- Only the classification head (GAP → BN → Dropout → Dense → Dense → Softmax) is trained
- LR: `1e-3` with Adam
- Purpose: Rapidly find a good weight initialization for the head without corrupting pretrained backbone weights
- EarlyStopping: patience=5 on `val_accuracy`

**Phase 2 — Selective Fine-Tuning:**
- `base_model.trainable = True`
- All layers except the last 40 are re-frozen: `for layer in base_model.layers[:-40]: layer.trainable = False`
- **BatchNormalization layers remain frozen throughout** — this is critical. BN layers contain running mean/variance statistics from ImageNet. Unfreezing them on a small dataset would corrupt these statistics and degrade transfer learning quality.
- LR: `1e-5` — two orders of magnitude lower than Phase 1 to prevent catastrophic forgetting
- EarlyStopping: patience=7 on `val_accuracy`
- `ModelCheckpoint` saves best model per fold based on `val_accuracy`

### Loss Function — Label Smoothing

```python
CategoricalCrossentropy(label_smoothing=0.1)
```

With 28 fine-grained classes where some motifs share visual similarities (e.g., `Solo_Parang` vs. `Yogyakarta_Parang`), hard labels (`[0, 0, 1, 0, ...]`) push the model toward extreme confidence. Label smoothing replaces hard `1.0` targets with `0.9`, distributing the remaining `0.1` uniformly across all other classes. This:
- Reduces overconfidence on similar-looking classes
- Improves calibration of the output probability distribution
- Acts as implicit regularization

### Class Weights

For each fold, class weights are recomputed from **that fold's training subset only** (not the full dataset) using `sklearn.utils.class_weight.compute_class_weight("balanced")`. This ensures that the weight calculation reflects the actual class distribution the model sees during training for that specific fold.

### Memory Management

After each fold completes, the model and base model are explicitly deleted, `tf.keras.backend.clear_session()` is called, and `gc.collect()` is run. This is essential on Kaggle — without this, GPU memory accumulates across folds and causes an OOM crash by fold 3 or 4.

---

## 8. Cell 5 — K-Fold Summary & Final Model

### Cross-Validation Report

After all 5 folds complete, results are printed as a table:
```
fold  val_accuracy  val_top3
   1        0.XXXX    0.XXXX
   ...
Mean Val Accuracy : X.XXXX ± X.XXXX
Mean Val Top-3    : X.XXXX ± X.XXXX
```

The standard deviation is the key metric — low std (< 0.03) confirms the model generalizes consistently regardless of which subset it trains on.

### Final Model Training (100% of Data)

Once K-Fold validates the architecture and hyperparameters, a **single final model is trained on all 85% trainval data** (no validation split). This maximizes the number of training samples for the production model.

Since there is no validation set, callbacks monitor **training metrics** (`accuracy`, `loss`) instead of validation metrics. This is intentional — we already know from K-Fold what the expected generalization is; the purpose of this run is to maximize weight quality for deployment.

The best checkpoint is saved to `MODEL_PATH` (`best_batik_model_v2.keras`).

---

## 9. Cell 6 — Evaluation with TTA

### Standard Evaluation

The final model is evaluated on the fixed 15% test holdout using `model.evaluate()`. This gives the baseline accuracy without augmentation at inference.

### Test-Time Augmentation (TTA)

TTA runs `TTA_STEPS=5` forward passes per test image, each with a different random light augmentation (same as Tier 3 pipeline). The softmax probability vectors from all passes are averaged before taking the argmax:

```
For each image in test set:
    For step in 1..5:
        Apply random Tier 3 augmentation
        Run model.predict → softmax vector
    Average the 5 softmax vectors
    argmax → final predicted class
```

**Why TTA works:** A single forward pass at test time sees one crop, one brightness, one hue. By averaging over 5 slightly different views of the same image, the model gets a more robust estimate of the true class probability — especially beneficial for edge cases and ambiguous textures.

The delta between standard accuracy and TTA accuracy is printed explicitly to quantify the TTA benefit.

### Outputs

- `test_classification_report.txt` — precision, recall, F1 per class (4 decimal places) + both accuracy figures
- `eval_confusion_matrix.png` — 28×28 heatmap, annotated with counts
- `eval_per_class_f1.png` — horizontal bar chart, color coded:  
  🔴 F1 < 0.70 | 🟠 F1 0.70–0.85 | 🟢 F1 ≥ 0.85

---

## 10. Cell 7 — TFLite Export

### The Keras 3 / Mixed Precision Deadlock

When Keras 3 (shipped with TF 2.16+) trains with `mixed_float16`, it injects dynamic `Cast` operations into the computational graph to shuttle values between float16 (for computation) and float32 (for optimizer updates). When `TFLiteConverter` traces this graph, it enters an infinite loop resolving these resource-typed variables — observable as an endless stream of `TensorSpec(shape=(), dtype=tf.resource, name=None)` lines with 0% CPU utilization.

### Fix: Float32 Graph Rebuild

The solution is to bypass the mixed-precision graph entirely:

1. Reset the global policy to `float32`
2. Rebuild an **identical but architecturally pure float32** EfficientNetV2S model (`weights=None`)
3. Load the trained weights from `MODEL_PATH` into this clean graph
4. Pass this float32 model to `TFLiteConverter.from_keras_model()` — no `Cast` ops, no deadlock

```python
tf.keras.mixed_precision.set_global_policy("float32")
# ... rebuild architecture identically ...
model_fp32.load_weights(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
```

### Quantization: FP16

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

FP16 quantization halves the model file size while maintaining near-identical accuracy (FP16 has 7 decimal digits of precision vs 15 for FP32 — more than sufficient for classification). INT8 quantization was considered but rejected because it requires a representative calibration dataset and risks accuracy degradation on fine-grained textures.

The conversion is wrapped in `with tf.device("/CPU:0")` to prevent any GPU interference during the conversion trace.

---

## 11. Cell 8 — Output Packaging

After all cells complete, Cell 8 creates a single ZIP archive containing all relevant output files (`.keras`, `.tflite`, `.txt`, `.csv`, `.png`). This allows downloading everything from the Kaggle Output panel in one click rather than individually selecting files.

---

## 12. Key Design Decisions

| Decision | Choice | Alternative Considered | Reason |
|---|---|---|---|
| Backbone | EfficientNetV2S | EfficientNetB0 | Higher capacity, faster convergence, better ImageNet accuracy |
| Validation strategy | 5-Fold Stratified K-Fold | Single random split | Small dataset makes single split statistically unreliable |
| Loss function | CategoricalCrossentropy (label_smoothing=0.1) | Hard labels | Prevents overconfidence on visually similar classes |
| Inference strategy | TTA (5 passes) | Single pass | Free accuracy gain, especially on ambiguous test images |
| Augmentation library | Albumentations 1.4.3 | tf.image | Richer transforms (ElasticTransform, GridDistortion) for textile textures |
| Augmentation strategy | Tier-based (3 tiers by count) | Uniform augmentation | Minority classes need stronger augmentation to compete |
| BatchNorm during fine-tune | Kept frozen | Unfreezing BN | Unfreezing BN on small dataset corrupts ImageNet statistics |
| TFLite quantization | FP16 | INT8 | Avoids calibration dataset requirement, minimal accuracy loss |
| TFLite conversion | Float32 graph rebuild | Direct from mixed-precision model | Avoids Keras 3 infinite trace deadlock |
| Deployment format | TFLite | ONNX, SavedModel | Optimal for Python CPU inference in Streamlit Community Cloud |
