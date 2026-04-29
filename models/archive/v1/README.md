# Model Archive — v1.0

This directory contains the original v1.0 deployment model and associated files,
preserved for reference and rollback purposes.

## How to restore v1.0

The full source code for v1.0 is permanently tagged in git:

```bash
git checkout v1.0
```

## v1.0 Specifications

| Property | Value |
|---|---|
| **Git tag** | `v1.0` |
| **Commit** | `e726165` |
| **Model file** | `batik_model_deploy.h5` |
| **Format** | Keras HDF5 (`.h5`) |
| **Backbone** | EfficientNetB0 (ImageNet pretrained) |
| **Classes** | 20 |
| **Input size** | 224 × 224 × 3 |
| **Reported test accuracy** | ~82% |
| **Model size** | ~18 MB |

## v1.0 Class List (20 classes)

```
Aceh_Pintu_Aceh
Bali_Barong
Bali_Merak
DKI_Ondel_Ondel
JawaBarat_Megamendung
JawaTimur_Pring
Kalimantan_Dayak
Lampung_Gajah
Madura_Mataketeran
Maluku_Pala
NTB_Lumbung
Papua_Asmat
Papua_Cendrawasih
Papua_Tifa
Solo_Parang
SulawesiSelatan_Lontara
SumateraBarat_Rumah_Minang
SumateraUtara_Boraspati
Yogyakarta_Kawung
Yogyakarta_Parang
```

## Why v2 was created

v1.0 was trained on a noisier dataset with geographic/technique labels that had
too much intra-class variance for a CNN to learn reliably. See the v2
`ARCHITECTURE.md` for the full data-centric rationale.
