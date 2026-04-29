import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf, cv2, gc
from sklearn.metrics import classification_report, confusion_matrix

print("Loading final model for evaluation...")
best_model = tf.keras.models.load_model(MODEL_PATH)

# Standard evaluation (no augmentation)
test_loss, test_acc, test_top3 = best_model.evaluate(test_ds, verbose=1)
print(f"\nStandard Eval -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Top-3: {test_top3:.4f}")

# Test-Time Augmentation (TTA_STEPS passes with light random augmentation)
print(f"\nRunning TTA ({TTA_STEPS} passes)...")

tta_preds = np.zeros((len(X_test), NUM_CLASSES))

for step in range(TTA_STEPS):
    tta_ds = build_dataset(X_test, y_test, tf_augment_tta, BATCH_SIZE, shuffle=False)
    tta_preds += best_model.predict(tta_ds, verbose=0)
    print(f"  TTA pass {step + 1}/{TTA_STEPS} done")

tta_preds /= TTA_STEPS
y_pred = np.argmax(tta_preds, axis=1)
y_true = y_test

tta_acc = np.mean(y_pred == y_true)
print(f"\nStandard accuracy : {test_acc:.4f}")
print(f"TTA accuracy      : {tta_acc:.4f}  (delta: {tta_acc - test_acc:+.4f})")

report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\nClassification Report:\n", report_str)

with open(os.path.join(OUTPUT_DIR, "test_classification_report.txt"), "w") as f:
    f.write(f"Standard Accuracy : {test_acc:.4f}\n")
    f.write(f"TTA Accuracy      : {tta_acc:.4f}\n\n")
    f.write(report_str)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Test Set (with TTA)", fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_confusion_matrix.png"), dpi=150)
plt.close()

# Per-class F1 bar chart
report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
f1_df = pd.DataFrame(
    [{"Class": cls, "F1_Score": report_dict[cls]["f1-score"]} for cls in CLASS_NAMES]
).sort_values(by="F1_Score")

colors = ["#d62728" if x < 0.70 else "#ff7f0e" if x < 0.85 else "#2ca02c" for x in f1_df["F1_Score"]]
plt.figure(figsize=(16, 10))
plt.barh(f1_df["Class"], f1_df["F1_Score"], color=colors, edgecolor="white")
plt.axvline(x=tta_acc, color="black", linestyle="--", label=f"TTA Accuracy ({tta_acc:.3f})")
plt.axvline(x=0.85, color="green", linestyle=":", alpha=0.6)
plt.axvline(x=0.70, color="red",   linestyle=":", alpha=0.6)
plt.title("Per-Class F1 Score (TTA Evaluation)")
plt.xlim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_per_class_f1.png"), dpi=120)
plt.close()

print("\nEvaluation Complete.")