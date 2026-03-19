import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="AI Deteksi Batik Nusantara",
    page_icon="👘",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #FFF8DC;
    }
    
    .stMarkdown, .stText, p, div, label, span, li {
        color: #4E342E !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #4E342E;
    }
    [data-testid="stSidebar"] * {
        color: #F5F5DC !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        color: #FFFFFF !important;
        background-color: #8B4513 !important;
        border-radius: 5px;
        border: 1px solid #DAA520;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #FFFFFF !important;
    }
    
    h1 {
        color: #8B4513 !important;
        font-family: 'Georgia', serif;
        text-align: center;
        border-bottom: 3px solid #8B4513;
        padding-bottom: 10px;
    }
    
    h2, h3, h4 {
        color: #A0522D !important;
        font-family: 'Georgia', serif;
    }
    
    .stButton>button {
        background-color: #8B4513;
        color: white !important;
        border-radius: 10px;
        border: 2px solid #DAA520;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #A0522D;
        border-color: #FFD700;
        color: white !important;
    }
    
    .batik-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        border-left: 10px solid #8B4513;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
        color: #333333 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #8B4513 !important;
        border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
         color: #4E342E !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
         background-color: #8B4513 !important;
         color: white !important;
         border: none;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

@st.cache_data
def load_labels():
    label_path = os.path.join(ROOT_DIR, 'models', 'labels.txt')
    
    if not os.path.exists(label_path):
        st.error(f"❌ File '{label_path}' tidak ditemukan.")
        st.warning("Pastikan file 'labels.txt' berada di direktori 'models'.")
        st.stop()
    
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

try:
    class_names = load_labels()
except Exception as e:
    st.error(f"Gagal membaca labels.txt: {e}")
    st.stop()

BATIK_INFO = {
    'Aceh_Pintu_Aceh': "Motif Pintu Aceh menggambarkan bentuk pintu rumah adat Aceh yang rendah. Filosofinya melambangkan kerendahan hati dan keterbukaan masyarakat Aceh, namun tetap memegang teguh batasan adat dan agama.",
    'Bali_Barong': "Motif ini terinspirasi dari Barong, makhluk mitologi Bali yang melambangkan kebajikan (Dharma). Motif ini sering digunakan dalam upacara sakral dan melambangkan perlindungan dari roh jahat.",
    'Bali_Merak': "Menggambarkan keindahan burung Merak yang melambangkan keanggunan dan kecantikan. Di Bali, motif ini sering dipadukan dengan ornamen floral yang dinamis.",
    'DKI_Ondel_Ondel': "Mengangkat ikon budaya Betawi, Ondel-ondel, yang dipercaya sebagai penolak bala. Motif ini biasanya menggunakan warna-warna cerah dan mencolok khas pesisir Jakarta.",
    'JawaBarat_Megamendung': "Ikon khas Cirebon ini berbentuk awan berarak dengan gradasi 7 warna. Melambangkan kesabaran dan kesejukan hati pemimpin (seperti awan yang meneduhkan) serta akulturasi budaya lokal dengan Tiongkok.",
    'Jawa_Barat_Megamendung': "Ikon khas Cirebon ini berbentuk awan berarak dengan gradasi 7 warna. Melambangkan kesabaran dan kesejukan hati pemimpin (seperti awan yang meneduhkan) serta akulturasi budaya lokal dengan Tiongkok.",
    'JawaTimur_Pring': "Motif Pring Sedapur (Rumpun Bambu) khas Magetan. Bambu melambangkan kerukunan dan persatuan (hidup bergerombol) serta ketenangan dan keteduhan.",
    'Jawa_Timur_Pring': "Motif Pring Sedapur (Rumpun Bambu) khas Magetan. Bambu melambangkan kerukunan dan persatuan (hidup bergerombol) serta ketenangan dan keteduhan.",
    'Kalimantan_Dayak': "Sering disebut Batik Batang Garing (Pohon Kehidupan). Motif ini melambangkan hubungan harmonis antara manusia, alam, dan Sang Pencipta dalam kepercayaan suku Dayak.",
    'Lampung_Gajah': "Gajah adalah hewan yang dihormati di Lampung (Way Kambas). Motif ini melambangkan kekuatan, kebesaran, dan kebijaksanaan, sering dipadukan dengan motif Siger (mahkota adat).",
    'Madura_Mataketeran': "Salah satu varian motif pesisir Madura dengan warna tajam (merah, kuning, hijau). Mata keteran sering diartikan sebagai mata perkutut, melambangkan kejelian dan keindahan.",
    'Maluku_Pala': "Terinspirasi dari kekayaan rempah Maluku, yaitu buah Pala. Motif ini melambangkan kemakmuran dan sejarah panjang Maluku sebagai pusat rempah dunia.",
    'NTB_Lumbung': "Menggambarkan Lumbung Padi (Uma Lengge) khas Bima atau Lombok. Melambangkan kesejahteraan, pangan yang melimpah, dan rasa syukur kepada Tuhan.",
    'Papua_Asmat': "Mengadopsi seni ukir suku Asmat yang geometris dan tegas. Sering menggambarkan patung leluhur (Mbis) yang melambangkan penghormatan kepada nenek moyang.",
    'Papua_Cendrawasih': "Menampilkan burung Cendrawasih (Bird of Paradise), fauna endemik Papua. Melambangkan keindahan abadi dan kebanggaan identitas masyarakat Papua.",
    'Papua_Tifa': "Menggambarkan alat musik Tifa. Melambangkan semangat persatuan, kegembiraan, dan denyut nadi kehidupan masyarakat Papua.",
    'Solo_Parang': "Salah satu motif larangan keraton. Bentuk 'S' jalin-menjalin melambangkan ombak samudera yang tak pernah putus (semangat pantang menyerah) dan jalinan silaturahmi.",
    'SulawesiSelatan_Lontara': "Mengangkat aksara Lontara kuno ke dalam kain. Melambangkan kecendekiaan, kearifan lokal, dan penghargaan terhadap ilmu pengetahuan warisan leluhur Bugis-Makassar.",
    'Sulawesi_Selatan_Lontara': "Mengangkat aksara Lontara kuno ke dalam kain. Melambangkan kecendekiaan, kearifan lokal, dan penghargaan terhadap ilmu pengetahuan warisan leluhur Bugis-Makassar.",
    'SumateraBarat_Rumah_Minang': "Menampilkan ikon Rumah Gadang dengan atap Bagonjong. Melambangkan perlindungan, kemegahan budaya Minangkabau, dan peran sentral keluarga.",
    'Sumatera_Barat_Rumah_Minang': "Menampilkan ikon Rumah Gadang dengan atap Bagonjong. Melambangkan perlindungan, kemegahan budaya Minangkabau, dan peran sentral keluarga.",
    'SumateraUtara_Boraspati': "Terinspirasi dari Boraspati Ni Tano (Cicak Tanah) dalam budaya Batak. Melambangkan kebijaksanaan, kekayaan bumi, dan pelindung rumah.",
    'Sumatera_Utara_Boraspati': "Terinspirasi dari Boraspati Ni Tano (Cicak Tanah) dalam budaya Batak. Melambangkan kebijaksanaan, kekayaan bumi, dan pelindung rumah.",
    'Yogyakarta_Kawung': "Bermotif geometris empat bulatan (buah aren/kolang-kaling). Melambangkan kesucian hati, pengendalian diri, dan harapan agar manusia berguna bagi sesamanya.",
    'Yogyakarta_Parang': "Varian Parang dari Yogyakarta (sering disebut Parang Rusak). Melambangkan perang melawan hawa nafsu dan keteguhan hati dalam membela kebenaran."
}

with st.sidebar:
    st.title("ℹ️ Tentang Projek")
    st.markdown("---")
    st.write("""
    **Aplikasi Klasifikasi Batik** ini dikembangkan untuk melestarikan dan memperkenalkan kekayaan motif Nusantara melalui kecerdasan buatan.
    """)
    
    st.subheader("🛠️ Teknologi")
    st.code("Python 3.10\nStreamlit\nTensorFlow (Keras)\nEfficientNetB0")
    
    st.subheader("📊 Performa Model")
    st.write("Akurasi Test: **82%**")
    st.write("Loss: **0.55**")
    
    st.info(f"Database memuat **{len(class_names)}** motif batik.")

    st.markdown("---")
    
    st.subheader("👥 Tim Pengembang")
    
    st.markdown("""
    <div style="font-size: 14px;">
        <ul style="padding-left: 20px; list-style-type: circle;">
            <li><strong>Benedict Michael Pepper</strong></li>
            <li><strong>Gilbetch Ronaldo Triswanto</strong></li>
            <li><strong>Sutri Ajeng Neng Rahayu</strong></li>
            <li><strong>Cecilia Margaretha</strong></li>
        </ul>
    </div>
    <div style="margin-top: 20px; font-size: 13px; text-align: center; border-top: 1px solid #F5F5DC; padding-top:10px;">
        <strong>Program Studi Teknik Informatika</strong><br>
        Universitas Ma Chung
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("© 2025 Projek PCD Batik")

@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, 'models', 'batik_model_deploy.h5')
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ File '{model_path}' tidak ditemukan.")
        return None
        
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {e}")
        return None

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    image_array = image_array.astype('float32')
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("AI Deteksi Motif Batik Nusantara")
st.markdown("*Unggah foto kain batik, dan biarkan AI mengungkap filosofinya.*")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Input Gambar")
    
    input_method = st.radio("Pilih Metode:", ["Upload File", "Gunakan Kamera"], horizontal=True)

    uploaded_file = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    else:
        camera_file = st.camera_input("Ambil foto kain batik")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Preview Gambar', use_column_width=True)
        
        predict_btn = st.button('🔍 IDENTIFIKASI MOTIF', use_container_width=True)

with col2:
    st.subheader("📝 Hasil Analisis")
    
    if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
        if model is None:
            st.error("Model gagal dimuat. Periksa file model .h5.")
        else:
            with st.spinner('Memproses gambar...'):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                predicted_class_idx = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                
                raw_label = class_names[predicted_class_idx]
                display_name = raw_label.replace("_", " ")
                description = BATIK_INFO.get(raw_label, "Informasi motif ini belum tersedia.")

                st.markdown(f"""
                <div class="batik-card">
                    <h2 style="margin-top:0; color:#8B4513 !important;">{display_name}</h2>
                    <p style="color:#333 !important;"><strong>Tingkat Keyakinan:</strong> {confidence:.2f}%</p>
                    <hr>
                    <p style="font-size:16px; line-height:1.6; color:#333 !important;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.caption("Kemungkinan motif lain:")
                top_5_idx = np.argsort(predictions[0])[-5:][::-1]
                top_5_probs = predictions[0][top_5_idx]
                top_5_names = [class_names[i].replace("_", " ") for i in top_5_idx]
                st.bar_chart(dict(zip(top_5_names, top_5_probs)), color="#8B4513")

    elif uploaded_file is None:
        st.info("Unggah gambar di panel kiri untuk memulai analisis.")