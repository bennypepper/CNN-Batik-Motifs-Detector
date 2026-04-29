import os
from pathlib import Path

dataset_dir = r"C:\Users\Benny Pepper\Documents\GitHub\CNN-Batik-Motifs-Detector\v2\dataset"
valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

records = []
for class_dir in sorted(Path(dataset_dir).iterdir()):
    if not class_dir.is_dir():
        continue
    images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
    records.append((class_dir.name, len(images)))

records.sort(key=lambda x: x[1])

total = sum(r[1] for r in records)
mean = total / len(records)

print(f"{'Class':<40} {'Count':>6}  {'Tier':>20}")
print("-" * 70)
for name, count in records:
    if count <= 45:
        tier = "Tier 1 (Aggressive)"
    elif count <= 90:
        tier = "Tier 2 (Moderate)"
    else:
        tier = "Tier 3 (Light)"
    flag = " *** LOW ***" if count < 20 else ""
    print(f"{name:<40} {count:>6}  {tier:>20}{flag}")

print("-" * 70)
print(f"{'TOTAL':<40} {total:>6}")
print(f"{'CLASSES':<40} {len(records):>6}")
print(f"{'MEAN per class':<40} {mean:>6.1f}")
print(f"{'MIN':<40} {min(r[1] for r in records):>6}")
print(f"{'MAX':<40} {max(r[1] for r in records):>6}")

t1 = [r for r in records if r[1] <= 45]
t2 = [r for r in records if 45 < r[1] <= 90]
t3 = [r for r in records if r[1] > 90]
print(f"\nTier 1 (<=45 imgs)  : {len(t1)} classes -> {[r[0] for r in t1]}")
print(f"Tier 2 (46-90 imgs) : {len(t2)} classes -> {[r[0] for r in t2]}")
print(f"Tier 3 (>90 imgs)   : {len(t3)} classes -> {[r[0] for r in t3]}")
print(f"\nFor 5-Fold, minimum samples per class must be >= 5")
print(f"Classes with < 5 images (cannot do 5-fold):")
low = [r for r in records if r[1] < 5]
print(f"  {low if low else 'None — all classes are safe for 5-fold!'}")
