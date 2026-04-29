import tensorflow as tf
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, gc

# K-Fold Cross-Validation Summary
results_df = pd.DataFrame(fold_results)

print("\n" + "=" * 55)
print("  K-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 55)
print(results_df.to_string(index=False))
print(f"\n  Mean Val Accuracy : {results_df['val_accuracy'].mean():.4f} ± {results_df['val_accuracy'].std():.4f}")
print(f"  Mean Val Top-3    : {results_df['val_top3'].mean():.4f} ± {results_df['val_top3'].std():.4f}")
print("=" * 55)

# Final model: train on 100% of trainval data using best hyperparams proven by K-Fold
print("\nFinal Model: Training on 100% of trainval data...")

class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_trainval)
final_class_weights = dict(enumerate(class_weights_arr))

full_train_ds = build_dataset(X_trainval, y_trainval, tf_augment_train, BATCH_SIZE, shuffle=True)

base_model, model = build_model(NUM_CLASSES)

# Phase 1: head only — monitored on training accuracy since no validation set
model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
)

print("  Phase 1: Training head...")
model.fit(
    full_train_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=[
        EarlyStopping(monitor="accuracy", patience=PHASE1_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1),
        CSVLogger(os.path.join(OUTPUT_DIR, "final_phase1_log.csv"), append=False),
    ],
    class_weight=final_class_weights,
    verbose=1,
)

# Phase 2: fine-tune
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
)

print("  Phase 2: Fine-tuning...")
model.fit(
    full_train_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=[
        ModelCheckpoint(filepath=MODEL_PATH, monitor="accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="accuracy", patience=PHASE2_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1),
        CSVLogger(os.path.join(OUTPUT_DIR, "final_phase2_log.csv"), append=False),
    ],
    class_weight=final_class_weights,
    verbose=1,
)

print(f"\nFinal model saved to: {MODEL_PATH}")