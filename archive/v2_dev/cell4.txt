import tensorflow as tf
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, gc

def build_model(num_classes: int) -> tuple:
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    base = EfficientNetV2S(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(DROPOUT_RATE_1, name="dropout_1")(x)
    x = Dense(DENSE_UNITS, activation="relu", name="dense_head")(x)
    x = Dropout(DROPOUT_RATE_2, name="dropout_2")(x)
    x = Dense(num_classes, name="dense_logits")(x)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs, name="batik_classifier_v2")
    return base, model

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1} / {N_FOLDS}")
    print(f"{'='*60}")

    X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
    fold_class_weights = dict(enumerate(class_weights_arr))

    train_ds_fold = build_dataset(X_tr, y_tr, tf_augment_train, BATCH_SIZE, shuffle=True)
    val_ds_fold   = build_dataset(X_val, y_val, tf_augment_eval, BATCH_SIZE, shuffle=False)

    base_model, model = build_model(NUM_CLASSES)

    # Phase 1: frozen backbone, train head only
    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    print(f"  Phase 1: Training head...")
    history1 = model.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=PHASE1_EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=PHASE1_PATIENCE, restore_best_weights=True, verbose=0, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=0),
        ],
        class_weight=fold_class_weights,
        verbose=1,
    )

    # Phase 2: unfreeze last N layers, keep BatchNorm frozen
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    fold_model_path = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_best.keras")

    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    print(f"  Phase 2: Fine-tuning last {UNFREEZE_LAYERS} layers...")
    history2 = model.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=PHASE2_EPOCHS,
        callbacks=[
            ModelCheckpoint(filepath=fold_model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
            EarlyStopping(monitor="val_accuracy", patience=PHASE2_PATIENCE, restore_best_weights=True, verbose=0, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=0),
        ],
        class_weight=fold_class_weights,
        verbose=1,
    )

    best_val_acc  = max(history2.history["val_accuracy"])
    best_val_top3 = max(history2.history["val_top3_accuracy"])

    fold_results.append({"fold": fold + 1, "val_accuracy": best_val_acc, "val_top3": best_val_top3})
    print(f"\n  Fold {fold + 1} -> val_acc: {best_val_acc:.4f} | val_top3: {best_val_top3:.4f}")

    del model, base_model, train_ds_fold, val_ds_fold
    tf.keras.backend.clear_session()
    gc.collect()