import streamlit as st
import os
import time
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AI Deteksi Batik Nusantara",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='50' fill='%238B4513'/><circle cx='50' cy='50' r='35' fill='%23FFF8DC'/><circle cx='50' cy='50' r='15' fill='%23C9962A'/></svg>",
    layout="wide"
)

def inject_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    # Remove blank lines to prevent Streamlit's markdown parser from breaking the <style> block
    css_clean = "\n".join([line for line in css.split('\\n') if line.strip() != ""])
    st.markdown(f"<style>{css_clean}</style>", unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@400;500;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inject_css(os.path.join(BASE_DIR, 'assets', 'style.css'))

from utils.data import REGION_MAP, WEAK_CLASSES, BATIK_INFO, load_labels
from utils.model_loader import load_model, preprocess_image
from components.sidebar import render_sidebar

class_names = load_labels()
model = load_model()

render_sidebar()

st.title("AI Deteksi Motif Batik Nusantara")
st.markdown("""
<div style="text-align:center; font-family:'Lato',sans-serif;
            font-size:1rem; color:#6B4226; margin: -12px 0 32px 0;
            font-style:italic;">
    Unggah foto kain batik — biarkan AI mengungkap nama, asal, dan filosofinya
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([11, 9])

with col1:
    st.markdown("<p style='font-size:1.1rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#8B4513; margin-bottom:12px; border-bottom:1px solid #E8C9A0; padding-bottom:6px;'>Input Gambar</p>", unsafe_allow_html=True)
    input_method = st.radio("Pilih Metode:", ["Upload File", "Gunakan Kamera"], horizontal=True, label_visibility="collapsed")

    uploaded_file = None
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        camera_file = st.camera_input("Ambil foto kain batik")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Preview', use_container_width=True)
        predict_btn = st.button('Identifikasi Motif', use_container_width=True)

with col2:
    st.markdown("<p style='font-size:1.1rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#8B4513; margin-bottom:12px; border-bottom:1px solid #E8C9A0; padding-bottom:6px;'>Hasil Analisis</p>", unsafe_allow_html=True)

    if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
        if model is None:
            st.error("Model gagal dimuat. Periksa file batik_model_v2.tflite.")
        else:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
                <div style="text-align:center; padding: 48px 24px;">
                    <svg style="animation: spin 3s linear infinite;" width="50" height="50" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                        <path d="M20 0c11 0 20 9 20 20s-9 20-20 20S0 31 0 20 9 0 20 0zm0 2c-10 0-18 8-18 18s8 18 18 18 18-8 18-18-8-18-18-18zm0 5c7 0 13 6 13 13s-6 13-13 13-13-6-13-13 6-13 13-13z" fill="#C9962A" fill-opacity="0.8"/>
                    </svg>
                    <p style="margin-top:24px; font-family:'Playfair Display', serif; color:#8B4513; font-size:1.15rem; font-style:italic;">"Setiap helai batik menyimpan cerita<br>yang menunggu untuk diungkap"</p>
                    <p style="font-size:0.85rem; color:#6B4226; margin-top:8px; letter-spacing:0.05em; text-transform:uppercase;">AI sedang menganalisis motif...</p>
                    <style>@keyframes spin { 100% { transform: rotate(360deg); } }</style>
                </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.1)

            processed_img = preprocess_image(image)
            input_details  = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], processed_img)
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])
            
            loading_placeholder.empty()

            predicted_class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions) * 100)
            raw_label    = class_names[predicted_class_idx]
            display_name = raw_label.replace("_", " ")
            description  = BATIK_INFO.get(raw_label, "Informasi motif ini belum tersedia.")
            region       = REGION_MAP.get(raw_label, "Indonesia")

            if confidence >= 70:
                bar_color = "#4A7C59"
                conf_label = "Tinggi"
            elif confidence >= 45:
                bar_color = "#C9962A"
                conf_label = "Cukup"
            else:
                bar_color = "#B74A2A"
                conf_label = "Rendah"

            bar_pct = min(confidence, 100)

            desc_sentences = description.split(". ")
            short_desc = desc_sentences[0] + "." if len(desc_sentences) > 1 else description[:120] + "..."

            st.markdown(f"""
            <div class="batik-card">
                <span class="region-badge">Asal: {region}</span>
                <div class="motif-title">{display_name}</div>
                <details class="motif-details">
                    <summary class="motif-summary">
                        <span class="short-desc">{short_desc}</span>
                        <div class="read-more-btn">Baca selengkapnya ▼</div>
                    </summary>
                    <div class="full-desc">{description}</div>
                </details>
                <div class="confidence-label">Keyakinan AI</div>
                <div class="confidence-bar-wrap">
                    <div class="confidence-bar-fill"
                         style="width:{bar_pct}%; background:{bar_color};"></div>
                </div>
                <div class="confidence-text">
                    {confidence:.1f}% &nbsp;&mdash;&nbsp;
                    <strong style="color:{bar_color};">{conf_label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if confidence < 45:
                st.warning("Keyakinan AI rendah. Coba foto lebih dekat, pencahayaan lebih baik, atau sudut berbeda.")
            elif raw_label in WEAK_CLASSES:
                st.warning(f"Motif **{display_name}** memiliki data pelatihan terbatas — interpretasikan hasil dengan hati-hati.")

            top5 = np.argsort(predictions[0])[-5:][::-1]
            top5_names = [class_names[i].replace("_", " ") for i in top5]
            top5_probs = predictions[0][top5]
            st.markdown("""
            <p style="font-size:1.1rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.05em; color:#8B4513; margin: 32px 0 16px 0;
                      border-bottom:1px solid #E8C9A0; padding-bottom:6px;">
                Kemungkinan Motif Lain
            </p>""", unsafe_allow_html=True)
            
            chart_html = "<div style='margin-top: 8px;'>"
            for n, p in zip(top5_names, top5_probs):
                pct = p * 100
                chart_html += f'<div style="margin-bottom: 12px;"><div style="display: flex; justify-content: space-between; font-family:\'Lato\',sans-serif; font-size: 0.9rem; color: #2C1A0E; margin-bottom: 4px;"><span>{n}</span><span style="font-weight: 700;">{pct:.1f}%</span></div><div style="width: 100%; background-color: #F5EDD4; border-radius: 6px; height: 10px; overflow: hidden;"><div style="width: {pct}%; background-color: #C9962A; border-radius: 6px; height: 100%; transition: width 0.6s ease;"></div></div></div>'
            chart_html += "</div>"
            st.markdown(chart_html, unsafe_allow_html=True)

    elif uploaded_file is None:
        st.markdown(f"""
        <div class="empty-state">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="40" cy="40" r="38" stroke="#E8C9A0" stroke-width="1.5" stroke-dasharray="4 3"/>
                <circle cx="40" cy="40" r="26" fill="#8B4513" opacity="0.08"/>
                <circle cx="40" cy="40" r="8" fill="#C9962A" opacity="0.7"/>
                <circle cx="40" cy="18" r="4" fill="#E8C9A0" opacity="0.6"/>
                <circle cx="40" cy="62" r="4" fill="#E8C9A0" opacity="0.6"/>
                <circle cx="18" cy="40" r="4" fill="#E8C9A0" opacity="0.6"/>
                <circle cx="62" cy="40" r="4" fill="#E8C9A0" opacity="0.6"/>
            </svg>
            <div class="empty-state-title">Unggah foto kain batik</div>
            <div class="empty-state-sub">
                Biarkan AI mengungkap nama, asal daerah,<br>dan makna filosofi di balik motifnya
            </div>
            <div class="empty-state-meta">
                84.7% akurasi &nbsp;·&nbsp; 28 motif batik Nusantara
            </div>
        </div>
        """, unsafe_allow_html=True)