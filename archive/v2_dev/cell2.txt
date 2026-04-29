import os, cv2
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def discover_dataset(root_dir: str) -> pd.DataFrame:
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No subdirectories found in: {root_dir}")

    for class_dir in class_dirs:
        label = class_dir.name
        for filepath in class_dir.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
                records.append({"filepath": str(filepath), "label": label})

    return pd.DataFrame(records)

df = discover_dataset(DATASET_DIR)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
CLASS_NAMES = list(label_encoder.classes_)
NUM_CLASSES = len(CLASS_NAMES)

with open(LABELS_PATH, "w") as f:
    f.write("\n".join(CLASS_NAMES))

print(f"Total images found : {len(df)}")
print(f"Total classes      : {NUM_CLASSES}")

corrupted, dimension_issues = [], []
for idx, row in df.iterrows():
    img = cv2.imread(row["filepath"])
    if img is None:
        corrupted.append(idx)
        continue
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        dimension_issues.append(idx)

if corrupted:
    df.drop(index=corrupted, inplace=True)
if dimension_issues:
    df.drop(index=dimension_issues, inplace=True)

df.reset_index(drop=True, inplace=True)
print(f"Clean dataset size : {len(df)} images")
print(f"Removed corrupted  : {len(corrupted)} | Sub-res: {len(dimension_issues)}")

class_counts = (
    df.groupby("label")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

class_counts["tier"] = class_counts["count"].apply(
    lambda x: "Tier 1 (Aggressive)" if x <= TIER1_THRESHOLD
    else ("Tier 2 (Moderate)" if x <= TIER2_THRESHOLD else "Tier 3 (Light)")
)

tier_colors = {
    "Tier 1 (Aggressive)": "#d62728",
    "Tier 2 (Moderate)":   "#ff7f0e",
    "Tier 3 (Light)":      "#2ca02c",
}

fig, ax = plt.subplots(figsize=(14, 12))
sorted_counts = class_counts.sort_values("count", ascending=True)
bar_colors = [tier_colors[t] for t in sorted_counts["tier"]]
bars = ax.barh(sorted_counts["label"], sorted_counts["count"], color=bar_colors, edgecolor="white", height=0.75)
for bar, count in zip(bars, sorted_counts["count"]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)
ax.axvline(x=TIER1_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
ax.axvline(x=TIER2_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.2)
ax.set_title("Class Distribution — Batik Dataset v2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_class_distribution.png"), dpi=150)
plt.close()

SAMPLES_PER_CLASS = 4
fig, axes = plt.subplots(NUM_CLASSES, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 2.2, NUM_CLASSES * 2.2))
for row_idx, class_name in enumerate(CLASS_NAMES):
    class_df = df[df["label"] == class_name]
    samples = class_df.sample(n=min(SAMPLES_PER_CLASS, len(class_df)), random_state=SEED)
    for col_idx in range(SAMPLES_PER_CLASS):
        ax = axes[row_idx, col_idx]
        if col_idx < len(samples):
            img = cv2.imread(samples.iloc[col_idx]["filepath"])
            if img is not None:
                ax.imshow(cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(class_name, fontsize=7, rotation=0, labelpad=80, va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_sample_grid.png"), dpi=120)
plt.close()