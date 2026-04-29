# CNN Batik Motifs Detector

An end-to-end deep learning pipeline and interactive web application that classifies **28 traditional Indonesian batik motifs** using a fine-tuned EfficientNetV2S backbone.

## Project Overview

Batik is a UNESCO-recognised intangible cultural heritage of Indonesia. Each region produces motifs with distinct visual signatures rooted in centuries of tradition. This project automates motif recognition to help preserve, document, and educate people about Indonesia's textile heritage.

### v2 Highlights

| Metric | v1 | v2 |
|---|---|---|
| Backbone | EfficientNetB0 | **EfficientNetV2S** |
| Classes | 20 | **28** |
| Validation | Single split | **5-Fold Stratified K-Fold** |
| Test Accuracy | ~82% | **84.68% (TTA)** |
| Macro F1 | 0.66 | **0.81** |
| Inference | Keras `.h5` | **TFLite FP16 (39 MB)** |

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** TensorFlow / Keras (EfficientNetV2S), TFLite for inference
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pillow, OpenCV, Pandas
- **Augmentation:** Albumentations (tier-based by class size)
- **Training Environment:** Kaggle (NVIDIA Tesla P100)

## Repository Structure

```
CNN-Batik-Motifs-Detector/
├── src/
│   └── app.py                  # Streamlit web application
├── models/
│   ├── batik_model_v2.tflite   # Production model (FP16 quantized)
│   ├── labels.txt              # 28 class names (alphabetical)
│   └── archive/v1/             # Archived v1 model + labels
├── notebooks/
│   ├── batik_motifs_v2_5.ipynb # Full Kaggle training notebook
│   └── cells/                  # Human-readable cell exports (.py)
├── results/v2.5/               # Evaluation outputs (plots, CSVs, reports)
├── docs/
│   ├── ARCHITECTURE_v2.md      # Full technical architecture reference
│   ├── code_review_v2.md       # Detailed code review
│   ├── laporan_project_akhir.pdf
│   └── presentasi_klasifikasi.pdf
├── scripts/                    # Dataset utility scripts
├── data/                       # Dataset directory (gitignored)
├── archive/v2_dev/             # Development scratch files
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Run Locally

```bash
git clone https://github.com/bennypepper/CNN-Batik-Motifs-Detector.git
cd CNN-Batik-Motifs-Detector
pip install -r requirements.txt
streamlit run src/app.py
```

> **Note:** The TFLite model (`models/batik_model_v2.tflite`, 39 MB) must be present. If git-ignored, download it from the [Releases](https://github.com/bennypepper/CNN-Batik-Motifs-Detector/releases) page.

### Deploy on Streamlit Community Cloud

1. Fork this repository
2. Upload `batik_model_v2.tflite` to `models/` (or use Git LFS)
3. Connect to [share.streamlit.io](https://share.streamlit.io)
4. Set main file path: `src/app.py`

## 28 Supported Batik Motifs

| # | Class | Region | # | Class | Region |
|---|---|---|---|---|---|
| 1 | Bali Barong | Bali | 15 | Papua Asmat | Papua |
| 2 | Bali Merak | Bali | 16 | Papua Cendrawasih | Papua |
| 3 | Ceplok | Jawa | 17 | Papua Tifa | Papua |
| 4 | Corak Insang | Kalimantan Barat | 18 | Priangan Merak Ngibing | Jawa Barat |
| 5 | Ikat Celup | Nusantara | 19 | Sekar | Jawa |
| 6 | Jakarta Ondel-Ondel | DKI Jakarta | 20 | Sidoluhur | Jawa |
| 7 | Jawa Barat Megamendung | Jawa Barat | 21 | Sogan | Solo |
| 8 | Jawa Timur Pring | Jawa Timur | 22 | Solo Parang | Solo |
| 9 | Kalimantan Dayak | Kalimantan | 23 | Sulawesi Selatan Lontara | Sulawesi |
| 10 | Lampung Gajah | Lampung | 24 | Sumatera Barat Rumah Minang | Sumatera |
| 11 | Lasem | Jawa Tengah | 25 | Sumatera Utara Boraspati | Sumatera |
| 12 | Madura Mataketeran | Madura | 26 | Tambal | Jawa |
| 13 | Maluku Pala | Maluku | 27 | Yogyakarta Kawung | Yogyakarta |
| 14 | NTB Lumbung | NTB | 28 | Yogyakarta Parang | Yogyakarta |

## Results

- **TTA Accuracy:** 84.68% (5-pass Test-Time Augmentation)
- **Macro F1:** 81.45%
- **Top performers:** Bali Barong, Madura Mataketeran, NTB Lumbung, Papua Tifa (F1 = 1.00)
- **Weak classes:** Priangan Merak Ngibing (F1 = 0.40), Sogan (F1 = 0.40), Lasem (F1 = 0.61)

For detailed per-class metrics, confusion matrix, and architecture decisions, see [`docs/ARCHITECTURE_v2.md`](docs/ARCHITECTURE_v2.md).

## Version History

| Version | Tag | Description |
|---|---|---|
| v1.0 | `v1.0` | EfficientNetB0, 20 classes, Keras `.h5`, ~82% accuracy |
| v2.0 | `main` | EfficientNetV2S, 28 classes, TFLite FP16, 84.68% TTA accuracy |

To restore v1: `git checkout v1.0`

## Team

- **Benedict Michael Pepper**
- **Gilbetch Ronaldo Triswanto**
- **Sutri Ajeng Neng Rahayu**
- **Cecilia Margaretha**

Program Studi Teknik Informatika — Universitas Ma Chung

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Dataset derived from [Batik-Indonesia](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia) by Muhammad Salman Al Faridzi (Apache 2.0).
