import albumentations as A
import tensorflow as tf
import numpy as np, cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Fixed 15% test holdout — never touched during K-Fold training
X_trainval, X_test, y_trainval, y_test = train_test_split(
    df["filepath"].values,
    df["label_encoded"].values,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label_encoded"].values,
)

print(f"Fixed test holdout : {len(X_test)} images")
print(f"Train+Val pool     : {len(X_trainval)} images")

# K-Fold splitter (applied to trainval pool in Cell 4)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Augmentation pipelines (tier-based)
AUGMENT_TIER1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.6),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
    A.ElasticTransform(alpha=80, sigma=8, p=0.3),
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=4, min_holes=1, max_height=32, min_height=16, max_width=32, min_width=16, p=0.3),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TIER2 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.4),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TIER3 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_VALIDATION = A.Compose([A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH)])

# TTA uses light random augmentation (same as Tier 3)
AUGMENT_TTA = AUGMENT_TIER3

def get_tier_for_class(class_name: str) -> str:
    count = df[df["label"] == class_name].shape[0]
    if count <= TIER1_THRESHOLD:
        return "tier1"
    elif count <= TIER2_THRESHOLD:
        return "tier2"
    return "tier3"

CLASS_TIER_MAP = {i: get_tier_for_class(CLASS_NAMES[i]) for i in range(NUM_CLASSES)}

def load_image(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath)
    if img is None:
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

def augment_train(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    label_int = int(label.numpy())
    tier = CLASS_TIER_MAP.get(label_int, "tier2")
    if tier == "tier1":
        img = AUGMENT_TIER1(image=img)["image"]
    elif tier == "tier2":
        img = AUGMENT_TIER2(image=img)["image"]
    else:
        img = AUGMENT_TIER3(image=img)["image"]
    return img.astype(np.float32), label_int

def augment_eval(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    img = AUGMENT_VALIDATION(image=img)["image"]
    return img.astype(np.float32), int(label.numpy())

def augment_tta(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    img = AUGMENT_TTA(image=img)["image"]
    return img.astype(np.float32), int(label.numpy())

def _wrap(aug_fn):
    def tf_fn(filepath, label):
        img, lbl = tf.py_function(func=aug_fn, inp=[filepath, label], Tout=[tf.float32, tf.int32])
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        lbl.set_shape([])
        return preprocess_input(img), tf.one_hot(lbl, depth=NUM_CLASSES)
    return tf_fn

tf_augment_train = _wrap(augment_train)
tf_augment_eval  = _wrap(augment_eval)
tf_augment_tta   = _wrap(augment_tta)

AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(filepaths, labels, augment_fn, batch_size, shuffle=False, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(AUTOTUNE)

# Build permanent test dataset (used in Cell 6)
test_ds = build_dataset(X_test, y_test, tf_augment_eval, BATCH_SIZE, shuffle=False, cache=True)

print("\nAugmentation tiers:")
for tier, classes in [("Tier 1", [CLASS_NAMES[i] for i, t in CLASS_TIER_MAP.items() if t == "tier1"]),
                       ("Tier 2", [CLASS_NAMES[i] for i, t in CLASS_TIER_MAP.items() if t == "tier2"]),
                       ("Tier 3", [CLASS_NAMES[i] for i, t in CLASS_TIER_MAP.items() if t == "tier3"])]:
    print(f"  {tier} ({len(classes)} classes): {classes}")