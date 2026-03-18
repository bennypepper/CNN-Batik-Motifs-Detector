# CNN Batik Motifs Detector 

An end-to-end Machine Learning pipeline and interactive web application that classifies 20 different traditional Indonesian Batik motifs using Deep Learning. 

## 📌 Project Overview
Batik is a highly complex traditional art form, making automated motif recognition challenging. This project successfully builds a classification model using **EfficientNetB0**, achieving an **81% accuracy rate** across 20 distinct classes. The model is deployed via a user-friendly Streamlit web interface, allowing users to upload images and get real-time predictions.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras (EfficientNetB0 architecture)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, OpenCV
* **Environment:** Jupyter Notebook, Kaggle

## 🚀 Key Features
* **Custom Transfer Learning Pipeline:** Utilized a 2-phase training approach (freezing/unfreezing layers) on the EfficientNetB0 base model to optimize feature extraction for complex patterns.
* **Robust Data Augmentation:** Handled dataset variance using rotation, zoom, and flipping techniques to prevent overfitting.
* **Interactive Deployment:** Built a lightweight Streamlit app (`app.py`) that handles image preprocessing and serves model predictions seamlessly.

## 📂 Repository Structure
```text
ai-batik-motifs-detector/
├── data/                   # Dataset samples
├── docs/                   # Final project reports and presentation slides
├── models/                 # Saved model weights (.h5) and label maps
├── notebooks/              # Jupyter notebooks containing model training and EDA
├── src/                    
│   └── app.py              # Streamlit web application code
├── requirements.txt        # Python dependencies
└── README.md
\`\`\`

## 💻 How to Run Locally

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/bennypepper/CNN-Batik-Motifs-Detector.git
   cd cnn-batik-motifs-detector
   \`\`\`

2. **Install the required dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Run the Streamlit application:**
   \`\`\`bash
   streamlit run src/app.py
   \`\`\`

## 📊 Results & Documentation
* The model achieved an F1-score of 0.81 on the test set. 
* For a deep dive into the confusion matrix, loss/accuracy curves, and architectural choices, please view the `docs/laporan_project_akhir.pdf`.
