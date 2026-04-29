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
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
    /* ═══════════════════════════════════════
       BATIK DESIGN TOKENS
    ═══════════════════════════════════════ */
    :root {
        --soga-900: #2C1A0E;
        --soga-700: #6B2D0F;
        --soga-500: #8B4513;
        --soga-300: #C4824A;
        --soga-100: #E8C9A0;
        --gold-500: #C9962A;
        --gold-300: #E8C94E;
        --surface-main: #FFF8DC;
        --surface-card: #FFFEF7;
        --surface-sidebar: #2C1A0E;
        --text-primary: #2C1A0E;
        --text-secondary: #6B4226;
        --text-on-dark: #F5EDD4;
        --warning-color: #C17D2A;
        --warning-bg: #FFF3D4;
        --warning-border: #E8A84A;
    }

    /* ═══════════════════════════════════════
       GLOBAL BASE
    ═══════════════════════════════════════ */
    .stApp {
        background-color: var(--surface-main);
        font-family: 'Lato', sans-serif;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    p, div, label, span, li {
        color: var(--text-primary);
        font-family: 'Lato', sans-serif;
        line-height: 1.65;
    }

    /* ═══════════════════════════════════════
       TYPOGRAPHY
    ═══════════════════════════════════════ */
    h1 {
        color: var(--soga-500) !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        text-align: center;
        border-bottom: 2px solid var(--soga-100);
        padding-bottom: 12px;
        letter-spacing: -0.01em;
    }
    h2 {
        color: var(--soga-500) !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    h3, h4 {
        color: var(--soga-300) !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    /* ═══════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: var(--surface-sidebar);
        background-image: repeating-linear-gradient(
            45deg,
            rgba(201,150,42,0.04) 0px,
            rgba(201,150,42,0.04) 1px,
            transparent 1px,
            transparent 12px
        );
    }
    [data-testid="stSidebar"] * {
        color: var(--text-on-dark) !important;
        font-family: 'Lato', sans-serif !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        font-family: 'Playfair Display', serif !important;
        color: var(--gold-500) !important;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--soga-100) !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        background-color: var(--soga-500) !important;
        border-radius: 6px;
        border: 1px solid var(--gold-500);
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #FFFFFF !important;
    }

    /* ═══════════════════════════════════════
       BUTTONS
    ═══════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, var(--soga-500) 0%, var(--soga-700) 100%);
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
        font-family: 'Lato', sans-serif !important;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.6rem 1.2rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(107, 45, 15, 0.35);
        background: linear-gradient(135deg, var(--soga-300) 0%, var(--soga-500) 100%);
        color: #FFFFFF !important;
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: none;
    }

    /* ═══════════════════════════════════════
       RESULT CARD
    ═══════════════════════════════════════ */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .batik-card {
        background-color: var(--surface-card);
        padding: 24px 28px;
        border-radius: 12px;
        border-top: 4px solid var(--soga-500);
        box-shadow: 0 4px 24px rgba(44, 26, 14, 0.10);
        margin-top: 16px;
        animation: fadeSlideIn 0.38s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .region-badge {
        display: inline-block;
        background: var(--soga-100);
        color: var(--soga-700) !important;
        font-family: 'Lato', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .motif-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--soga-500) !important;
        margin: 4px 0 12px 0;
        line-height: 1.2;
    }
    .motif-desc {
        font-family: 'Lato', sans-serif;
        font-size: 0.97rem;
        line-height: 1.7;
        color: var(--text-primary) !important;
    }
    .confidence-label {
        font-family: 'Lato', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary) !important;
        margin-top: 20px;
        margin-bottom: 4px;
    }
    .confidence-bar-wrap {
        background: var(--soga-100);
        border-radius: 99px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-bottom: 4px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }
    .confidence-text {
        font-size: 0.8rem;
        font-family: 'Lato', sans-serif;
        color: var(--text-secondary) !important;
    }

    /* ═══════════════════════════════════════
       WARNING — OVERRIDE STREAMLIT YELLOW
    ═══════════════════════════════════════ */
    [data-testid="stAlert"][kind="warning"],
    div[data-testid="stAlert"] {
        background-color: var(--warning-bg) !important;
        border-left-color: var(--warning-color) !important;
        color: var(--soga-700) !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: var(--soga-700) !important;
    }

    /* ═══════════════════════════════════════
       FILE UPLOADER
    ═══════════════════════════════════════ */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed var(--soga-300) !important;
        border-radius: 10px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--soga-500) !important;
    }
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: var(--text-secondary) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--soga-500) !important;
        color: white !important;
        border: none;
        border-radius: 6px !important;
    }

    /* ═══════════════════════════════════════
       EMPTY STATE
    ═══════════════════════════════════════ */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        opacity: 0.85;
    }
    .empty-state-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: var(--soga-500) !important;
        margin: 16px 0 8px 0;
    }
    .empty-state-sub {
        font-size: 0.875rem;
        color: var(--text-secondary) !important;
        line-height: 1.5;
    }
    .empty-state-meta {
        margin-top: 20px;
        font-size: 0.78rem;
        color: var(--soga-100) !important;
        background: var(--soga-700);
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
    }

    /* ═══════════════════════════════════════
       SIDEBAR INFO CARDS
    ═══════════════════════════════════════ */
    .sidebar-stat {
        background: rgba(201,150,42,0.12);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid var(--gold-500);
    }
    .sidebar-stat-val {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--gold-500) !important;
    }
    .sidebar-stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--soga-100) !important;
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
    'Bali_Barong': "Batik Barong Bali terinspirasi dari Barong, makhluk mitologi Bali yang melambangkan kekuatan kebajikan (Dharma) dalam pertarungan abadi melawan Rangda (Adharma). Motif ini menampilkan figur singa bermahkota dengan detail ornamen yang rumit, sering dipadukan dengan motif api dan sulur-suluran khas seni ukir Bali. Batik Barong memiliki nilai sakral dan umumnya digunakan dalam upacara keagamaan Hindu-Bali sebagai simbol perlindungan spiritual dari roh jahat dan energi negatif.",
    'Bali_Merak': "Batik Merak Bali menggambarkan keindahan burung Merak yang sedang mengembangkan ekor indahnya, melambangkan keanggunan, kecantikan, dan kemewahan. Dalam tradisi Bali, motif ini sering dipadukan dengan ornamen floral seperti bunga kamboja dan dedaunan tropis yang dinamis, menciptakan komposisi yang penuh warna dan hidup. Batik ini populer digunakan dalam busana pesta dan upacara adat karena kesan mewah dan anggunnya, serta menjadi salah satu motif favorit wisatawan sebagai oleh-oleh khas Bali.",
    'Ceplok': "Motif Ceplok merupakan salah satu pola batik tertua di Jawa, ditandai dengan susunan geometris yang berulang secara simetris dalam bentuk kotak, lingkaran, roset, atau bintang. Nama 'Ceplok' berasal dari kata Jawa yang merujuk pada potongan buah yang tampak simetris saat dibelah. Motif ini melambangkan keteraturan alam semesta, keseimbangan hidup, dan harmoni antara manusia dengan lingkungannya. Ceplok memiliki ratusan variasi seperti Ceplok Kasatrian, Ceplok Sriwedari, dan Ceplok Grompol, masing-masing dengan makna filosofis yang berbeda sesuai daerah asalnya.",
    'Corak_Insang': "Batik Corak Insang adalah motif khas Melayu Pontianak, Kalimantan Barat, yang mengambil inspirasi dari bentuk insang dan sisik ikan sebagai cerminan kehidupan masyarakat pesisir sungai Kapuas. Pola garis-garis paralel yang menyerupai insang ikan disusun secara berulang dengan variasi warna cerah seperti merah, kuning, dan hijau. Motif ini melambangkan keharmonisan masyarakat Melayu dengan ekosistem sungai dan laut yang menjadi sumber kehidupan mereka, serta mengekspresikan harapan akan rezeki yang melimpah seperti ikan di sungai.",
    'Ikat_Celup': "Batik Ikat Celup (juga dikenal sebagai tie-dye batik) merupakan teknik pembuatan batik yang menciptakan pola organik dan abstrak melalui proses mengikat bagian-bagian kain dengan tali atau karet sebelum mencelupkannya ke dalam pewarna. Berbeda dengan batik tulis atau cap yang menggunakan malam (lilin), teknik ikat celup menghasilkan gradasi warna yang mengalir bebas dan pola yang tidak dapat direplikasi secara persis. Setiap lembar kain ikat celup bersifat unik, menjadikannya karya seni tekstil satu-satunya di dunia. Teknik ini tersebar luas di seluruh Nusantara dan sangat populer di kalangan pengrajin muda.",
    'Jakarta_Ondel_Ondel': "Batik Ondel-ondel mengangkat ikon budaya Betawi yang paling dikenal, yaitu boneka raksasa Ondel-ondel yang dipercaya sebagai penolak bala dan pelindung kampung dari gangguan roh halus. Motif ini menampilkan sepasang figur Ondel-ondel laki-laki dan perempuan dengan wajah khas berwarna merah dan putih, sering dikelilingi ornamen kembang kelapa dan kembang goyang. Batik ini menggunakan palet warna cerah dan mencolok khas pesisir Jakarta seperti merah, oranye, dan hijau terang, mencerminkan karakter masyarakat Betawi yang terbuka, ceria, dan ekspresif.",
    'Jawa_Barat_Megamendung': "Batik Megamendung adalah ikon khas Cirebon, Jawa Barat, yang menampilkan motif awan berarak dengan gradasi tujuh lapisan warna dari gelap ke terang. Gradasi ini melambangkan tujuh lapisan langit dalam kosmologi lokal serta tahapan menuju kebijaksanaan spiritual. Secara filosofis, Megamendung mengajarkan kesabaran dan kesejukan hati seorang pemimpin yang harus mampu meneduhkan rakyatnya seperti awan yang melindungi bumi dari terik matahari. Motif ini merupakan hasil akulturasi budaya Jawa dengan estetika Tiongkok yang dibawa oleh pedagang dari daratan Cina pada abad ke-14.",
    'Jawa_Timur_Pring': "Batik Pring Sedapur (Rumpun Bambu) adalah motif khas Magetan, Jawa Timur, yang menampilkan rumpun bambu dengan daun-daun yang melambai tertiup angin. Nama 'Pring Sedapur' secara harfiah berarti 'serumpun bambu', menggambarkan sifat bambu yang selalu tumbuh berkelompok dan tidak pernah berdiri sendiri. Filosofinya melambangkan kerukunan, gotong-royong, dan persatuan masyarakat yang hidup berdampingan secara harmonis. Bambu juga dikenal sebagai tanaman yang lentur namun kuat, sehingga motif ini turut melambangkan ketahanan dan ketenangan dalam menghadapi badai kehidupan.",
    'Kalimantan_Dayak': "Batik Dayak menampilkan motif Batang Garing (Pohon Kehidupan) yang merupakan simbol kosmologi suku Dayak di Kalimantan. Pohon Kehidupan digambarkan sebagai pohon sakral yang akarnya menghujam ke dunia bawah dan cabangnya menjulang ke langit, menghubungkan tiga alam: dunia bawah, dunia manusia, dan dunia atas. Motif ini melambangkan hubungan harmonis antara manusia, alam, dan Sang Pencipta (Ranying Hatalla) dalam kepercayaan Kaharingan. Batik Dayak sering menggunakan warna-warna alam seperti cokelat, hitam, dan merah bata yang diambil dari pewarna alami hutan Kalimantan.",
    'Lampung_Gajah': "Batik Gajah Lampung menggambarkan gajah Sumatera yang merupakan hewan kebanggaan provinsi Lampung, terutama terkait dengan Taman Nasional Way Kambas sebagai pusat konservasi gajah. Motif gajah biasanya ditampilkan dengan detail yang megah, lengkap dengan ornamen kerajaan dan sering dipadukan dengan motif Siger, yaitu mahkota adat emas berbentuk tanduk kerbau yang menjadi simbol kehormatan wanita Lampung. Secara filosofis, motif ini melambangkan kekuatan, kebesaran, kebijaksanaan, dan kemakmuran, serta menjadi pengingat akan pentingnya menjaga kelestarian satwa langka Sumatera.",
    'Lasem': "Batik Lasem berasal dari kota Lasem di pesisir utara Jawa Tengah, yang dijuluki 'Tiongkok Kecil' karena sejarah panjang komunitas Tionghoa di sana sejak abad ke-15. Batik ini sangat dikenal dengan warna merah tua khasnya yang disebut 'abang getih pithik' (merah darah ayam), sebuah warna yang sulit ditiru oleh daerah lain karena menggunakan resep pewarna alami turun-temurun. Motif-motifnya menggabungkan estetika Jawa dan Tionghoa, menampilkan naga, burung hong, bunga seruni, dan flora-fauna laut yang kaya detail. Batik Lasem merupakan bukti nyata akulturasi budaya yang harmonis dan telah diakui sebagai salah satu warisan budaya tak benda Indonesia.",
    'Madura_Mataketeran': "Batik Mataketeran adalah motif khas pesisir Madura yang terkenal dengan penggunaan warna-warna tajam dan berani seperti merah menyala, kuning cerah, dan hijau tua yang mencerminkan karakter masyarakat Madura yang tegas dan ekspresif. Nama 'Mataketeran' sering diartikan sebagai mata burung perkutut, burung yang sangat dihargai dalam budaya Jawa dan Madura sebagai simbol kejelian, keindahan suara, dan keberuntungan. Motif ini menampilkan pola mata burung yang disusun berulang dengan ornamen daun dan bunga di sekelilingnya, dan sering digunakan dalam acara-acara adat Madura seperti karapan sapi dan upacara pernikahan.",
    'Maluku_Pala': "Batik Pala Maluku terinspirasi dari buah pala (Myristica fragrans), rempah legendaris yang menjadikan Kepulauan Maluku sebagai pusat perdagangan dunia selama berabad-abad dan pernah menjadi rebutan bangsa-bangsa Eropa. Motif ini menampilkan buah pala lengkap dengan fuli (selaput buah) berwarna merah yang menyelimuti biji, disusun secara dekoratif bersama daun dan ranting pohon pala. Secara filosofis, batik Pala melambangkan kemakmuran, kekayaan alam, dan kebanggaan sejarah Maluku sebagai 'Kepulauan Rempah' yang pernah menguasai perekonomian maritim global.",
    'NTB_Lumbung': "Batik Lumbung menggambarkan bangunan Lumbung Padi tradisional yang dikenal sebagai Uma Lengge di Bima atau Sambi di Lombok, sebuah struktur penyimpanan padi khas Nusa Tenggara Barat dengan atap berbentuk kerucut tinggi yang menjulang. Lumbung padi bukan sekadar tempat penyimpanan, melainkan simbol kesejahteraan, kemakmuran, dan ketahanan pangan masyarakat agraris NTB. Motif ini melambangkan rasa syukur kepada Tuhan atas hasil bumi yang melimpah, serta mengingatkan pentingnya menyimpan dan mengelola sumber daya dengan bijaksana untuk masa depan.",
    'Papua_Asmat': "Batik Papua Asmat mengadopsi seni ukir suku Asmat yang terkenal di seluruh dunia dengan pola geometris yang tegas, simetris, dan penuh makna spiritual. Motif ini sering menggambarkan patung leluhur (Mbis), topeng ritual, dan simbol-simbol alam seperti buaya, burung kasuari, dan pohon sagu yang menjadi sumber kehidupan masyarakat Asmat. Setiap garis dan pola dalam ukiran Asmat memiliki makna sakral yang berkaitan dengan penghormatan kepada arwah nenek moyang dan siklus kehidupan-kematian. Batik Asmat biasanya menggunakan palet warna earthy seperti cokelat, hitam, putih, dan merah tanah.",
    'Papua_Cendrawasih': "Batik Cendrawasih menampilkan burung Cendrawasih (Bird of Paradise), fauna endemik Papua yang terkenal dengan bulu-bulu ekornya yang panjang, berwarna-warni, dan memukau saat melakukan tarian kawin. Burung ini dianggap sebagai jelmaan dewata oleh masyarakat Papua dan menjadi simbol keindahan abadi serta kebanggaan identitas budaya Papua. Motif batik Cendrawasih biasanya menampilkan burung jantan dengan ekor mekar penuh dalam pose menari, dikelilingi oleh dedaunan tropis dan bunga anggrek hutan. Batik ini menjadi salah satu motif paling ikonik dari Papua dan sering digunakan dalam acara-acara resmi kenegaraan.",
    'Papua_Tifa': "Batik Tifa menggambarkan alat musik tradisional Tifa, sejenis gendang kayu panjang yang dilubangi dan ditutup kulit binatang pada salah satu ujungnya. Tifa merupakan instrumen sakral yang tidak terpisahkan dari kehidupan masyarakat Papua dan Maluku, digunakan dalam upacara adat, perayaan, tarian perang, dan penyambutan tamu kehormatan. Motif batik ini menampilkan Tifa dalam berbagai ukuran dan posisi, sering dipadukan dengan ornamen tribal dan pola-pola geometris khas Papua. Secara filosofis, Tifa melambangkan semangat persatuan, kegembiraan komunal, dan denyut nadi kehidupan masyarakat Papua yang selalu bersemangat.",
    'Priangan_Merak_Ngibing': "Motif Merak Ngibing ('Merak yang Menari') merupakan kebanggaan batik Priangan, Jawa Barat, yang menggambarkan sepasang burung merak jantan dan betina sedang menari berhadapan dalam pose yang anggun dan simetris. Kata 'Ngibing' dalam bahasa Sunda berarti menari, merujuk pada gerakan merak jantan yang mengembangkan ekor indahnya untuk memikat pasangan. Motif ini melambangkan keindahan, keanggunan, cinta kasih, dan romantisme dalam kehidupan berumah tangga. Batik Merak Ngibing biasanya dikerjakan dengan teknik batik tulis halus menggunakan warna-warna lembut khas Priangan seperti biru indigo, cokelat soga, dan krem.",
    'Sekar': "Sekar atau Sekar Jagad berasal dari kata Jawa 'sekar' yang berarti bunga atau keindahan, dan 'jagad' yang berarti dunia. Motif ini menampilkan berbagai stilisasi bunga dan tumbuhan yang disusun dalam kompartemen-kompartemen organik yang saling berdampingan, menciptakan kesan peta keindahan dunia. Setiap kompartemen berisi motif yang berbeda, melambangkan keberagaman budaya dan keindahan alam semesta yang hidup berdampingan secara harmonis. Batik Sekar sering digunakan dalam upacara pernikahan Jawa sebagai simbol keindahan, kesuburan, kegembiraan hidup, dan harapan akan kehidupan yang penuh berkah.",
    'Sidoluhur': "Batik Sidoluhur termasuk dalam kelompok motif 'Sido' yang bermakna 'jadi' atau 'menjadi', sementara 'Luhur' berarti mulia atau agung. Motif ini merupakan salah satu batik larangan keraton Surakarta dan Yogyakarta yang dahulu hanya boleh dikenakan oleh keluarga kerajaan dan bangsawan tinggi. Pola Sidoluhur menampilkan susunan geometris berisi ornamen gurda (sayap garuda), meru (gunung), dan berbagai simbol kemuliaan yang disusun simetris dalam kotak-kotak. Filosofinya merupakan doa dan harapan agar pemakainya menjadi pribadi yang luhur budinya, berwibawa, dihormati, dan selalu berada di jalan kebenaran.",
    'Sogan': "Batik Sogan adalah gaya batik khas Surakarta (Solo) yang dikenali dari warna cokelat keemasan hangat yang berasal dari pewarna alami kulit pohon soga (Peltophorum pterocarpum). Warna sogan dianggap sebagai warna paling elegan dan sopan dalam tradisi batik Jawa, berbeda dari batik pesisir yang cenderung berwarna cerah. Motif-motif Sogan biasanya berupa pakem klasik keraton seperti Truntum, Sidomukti, atau Parang yang dikerjakan dengan teknik batik tulis halus dan penuh kesabaran. Batik Sogan mencerminkan kepribadian masyarakat Solo yang halus, penuh tata krama, dan menjunjung tinggi kehalusan budi pekerti dalam setiap aspek kehidupan.",
    'Solo_Parang': "Batik Parang Solo merupakan salah satu motif larangan keraton tertua dan paling prestisius dalam tradisi batik Jawa, yang dahulu hanya boleh dikenakan oleh raja dan keluarga kerajaan Kasunanan Surakarta. Bentuk utama motif Parang adalah huruf 'S' yang saling jalin-menjalin tanpa putus secara diagonal, terinspirasi dari ombak samudera yang terus bergerak tanpa henti. Filosofinya melambangkan semangat pantang menyerah, kegigihan dalam menghadapi tantangan hidup, serta jalinan silaturahmi yang tidak boleh terputus antar generasi. Variasi Parang sangat beragam, mulai dari Parang Barong (paling besar, khusus raja) hingga Parang Klithik (lebih kecil, untuk pejabat).",
    'Sulawesi_Selatan_Lontara': "Batik Lontara Sulawesi Selatan mengangkat aksara Lontara, sistem tulisan kuno suku Bugis-Makassar yang berbentuk kotak-kotak dengan garis melengkung di dalamnya, ke dalam motif kain yang artistik dan bermakna. Aksara Lontara digunakan untuk menulis naskah-naskah kuno seperti La Galigo, salah satu karya sastra terpanjang di dunia yang menceritakan mitologi Bugis. Motif ini melambangkan kecendekiaan, penghargaan terhadap ilmu pengetahuan, dan kebanggaan akan warisan intelektual leluhur Bugis-Makassar yang telah mengembangkan sistem navigasi, hukum maritim, dan perdagangan jauh sebelum era kolonial.",
    'Sumatera_Barat_Rumah_Minang': "Batik Rumah Minang menampilkan ikon arsitektur paling terkenal dari Sumatera Barat, yaitu Rumah Gadang dengan atap Bagonjong yang menjulang runcing menyerupai tanduk kerbau. Rumah Gadang bukan sekadar tempat tinggal, melainkan simbol identitas suku Minangkabau yang menganut sistem matrilineal, di mana rumah menjadi pusat kehidupan keluarga besar dari garis keturunan ibu. Motif ini melambangkan perlindungan, kemegahan budaya, kehangatan kekeluargaan, dan peran sentral perempuan Minang dalam menjaga adat dan tradisi. Batik Rumah Minang sering dipadukan dengan ornamen ukiran khas Minang yang disebut 'Aka Bapilin' (akar berpilin).",
    'Sumatera_Utara_Boraspati': "Batik Boraspati terinspirasi dari Boraspati Ni Tano (Cicak Tanah), makhluk mitologis dalam kepercayaan suku Batak Toba yang digambarkan menyerupai kadal atau cicak dan dipercaya sebagai penjaga bumi dan pelindung rumah. Boraspati Ni Tano dianggap sebagai roh penguasa tanah yang menjaga kesuburan dan melindungi penghuni rumah dari marabahaya. Motif ini biasanya menampilkan figur cicak yang distilisasi secara geometris bersama ornamen Gorga (ukiran tradisional Batak) yang penuh warna. Secara filosofis, batik Boraspati melambangkan kebijaksanaan, kekayaan bumi, perlindungan spiritual, dan hubungan sakral antara manusia dengan tanah leluhurnya.",
    'Tambal': "Batik Tambal memiliki tampilan visual yang unik karena terdiri dari potongan-potongan berbagai motif batik yang berbeda, dijahit atau digambar menjadi satu lembar kain seperti sebuah quilt atau tambal sulam. Kata 'Tambal' secara harfiah berarti menambal atau memperbaiki sesuatu yang rusak. Dalam tradisi Jawa kuno, kain batik Tambal digunakan sebagai selimut penyembuhan yang diselimutkan kepada orang sakit dengan harapan mempercepat kesembuhan melalui energi gabungan dari berbagai motif yang masing-masing membawa doa dan harapan tersendiri. Filosofi batik Tambal mengajarkan bahwa keindahan dan kekuatan justru lahir dari keberagaman yang disatukan.",
    'Yogyakarta_Kawung': "Batik Kawung merupakan salah satu motif batik tertua di Jawa yang telah ditemukan pada relief Candi Prambanan abad ke-9. Motifnya terdiri dari empat bulatan lonjong yang tersusun secara geometris menyerupai irisan buah aren (kolang-kaling) atau bunga teratai dilihat dari atas. Kawung termasuk motif larangan keraton Yogyakarta yang dahulu hanya diperuntukkan bagi sultan dan abdi dalem keraton. Secara filosofis, empat bulatan Kawung melambangkan empat arah mata angin dan empat unsur kehidupan, mengajarkan kesucian hati, pengendalian diri dari hawa nafsu, serta harapan agar manusia senantiasa berguna bagi sesama layaknya buah aren yang seluruh bagiannya bermanfaat.",
    'Yogyakarta_Parang': "Batik Parang Yogyakarta, sering disebut Parang Rusak, merupakan motif larangan keraton Kesultanan Yogyakarta yang memiliki makna filosofis mendalam tentang perjuangan batin manusia. Kata 'Rusak' di sini bukan berarti hancur, melainkan merujuk pada kata 'tebing batu karang yang terjal' (parang = tebing), menggambarkan kekuatan alam yang dahsyat namun kokoh. Pola diagonal yang tak terputus melambangkan semangat perang melawan hawa nafsu dan godaan duniawi, serta keteguhan hati dalam membela kebenaran dan keadilan. Parang Rusak Barong merupakan varian terbesar dan paling agung yang khusus dikenakan Sultan, sementara varian yang lebih kecil diperuntukkan bagi kerabat keraton sesuai tingkatan pangkatnya."
}


with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 8px 0 16px 0;">
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                        font-weight:700; color:#C9962A; letter-spacing:-0.01em;">
                AI Deteksi Batik
            </div>
            <div style="font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; color:#E8C9A0; margin-top:2px;">
                Nusantara
            </div>
        </div>
        <hr style="border-color:rgba(201,150,42,0.25); margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    # ── Cara Penggunaan ──
    with st.expander("Cara Penggunaan", expanded=True):
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.7; color:#F5EDD4;">
            <b style="color:#C9962A;">1.</b> Unggah foto kain batik atau gunakan kamera<br>
            <b style="color:#C9962A;">2.</b> Klik tombol <em>Identifikasi Motif</em><br>
            <b style="color:#C9962A;">3.</b> Temukan nama, asal, dan filosofi motif
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Model Stats ──
    st.markdown("""
        <div class="sidebar-stat">
            <div class="sidebar-stat-val">84.7%</div>
            <div class="sidebar-stat-label">Akurasi TTA (Test-Time Aug.)</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-val">81.45%</div>
            <div class="sidebar-stat-label">Macro F1-Score</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-val">28</div>
            <div class="sidebar-stat-label">Motif Batik Nusantara</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── About ──
    with st.expander("Tentang Proyek"):
        st.markdown("""
        <div style="font-size:0.83rem; line-height:1.6; color:#F5EDD4;">
            Aplikasi ini dikembangkan untuk melestarikan dan memperkenalkan
            kekayaan motif batik Nusantara melalui kecerdasan buatan berbasis
            <em>EfficientNetV2S</em> dengan 5-Fold Cross-Validation.
        </div>
        """, unsafe_allow_html=True)

    # ── Tim ──
    with st.expander("Tim Pengembang"):
        st.markdown("""
        <div style="font-size:0.83rem; line-height:1.9; color:#F5EDD4;">
            Benedict Michael Pepper<br>
            Gilbetch Ronaldo Triswanto<br>
            Sutri Ajeng Neng Rahayu<br>
            Cecilia Margaretha
        </div>
        <div style="margin-top:10px; font-size:0.75rem; color:#E8C9A0;
                    border-top:1px solid rgba(232,201,160,0.2); padding-top:8px;">
            Program Studi Teknik Informatika<br>Universitas Ma Chung
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <hr style="border-color:rgba(201,150,42,0.2); margin:16px 0 8px 0;">
        <div style="text-align:center; font-size:0.72rem; color:#E8C9A0;">
            &copy; 2026 Projek PCD Batik
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, 'models', 'batik_model_v2.tflite')
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        return None

model = load_model()

def preprocess_image(image):
    """
    Preprocess image for the TFLite model.

    The training pipeline applied preprocess_input() externally (scaling
    [0, 255] → [-1, 1]) before feeding into EfficientNetV2S which also has
    The TFLite model contains an internal rescaling_1_1 layer
    (include_preprocessing=True in the rebuilt graph) that maps [0, 255] → [-1, 1].
    Diagnostic confirmed: raw [0-255] input produces strongest signal.
    Do NOT apply external preprocess_input() — let the model handle it.
    """
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return np.expand_dims(image_array.astype('float32'), axis=0)

REGION_MAP = {
    'Bali_Barong': 'Bali', 'Bali_Merak': 'Bali',
    'Ceplok': 'Jawa Tengah', 'Corak_Insang': 'Kalimantan Barat',
    'Ikat_Celup': 'Nusantara', 'Jakarta_Ondel_Ondel': 'DKI Jakarta',
    'Jawa_Barat_Megamendung': 'Jawa Barat', 'Jawa_Timur_Pring': 'Jawa Timur',
    'Kalimantan_Dayak': 'Kalimantan', 'Lampung_Gajah': 'Lampung',
    'Lasem': 'Jawa Tengah', 'Madura_Mataketeran': 'Madura',
    'Maluku_Pala': 'Maluku', 'NTB_Lumbung': 'Nusa Tenggara Barat',
    'Papua_Asmat': 'Papua', 'Papua_Cendrawasih': 'Papua',
    'Papua_Tifa': 'Papua', 'Priangan_Merak_Ngibing': 'Jawa Barat',
    'Sekar': 'Jawa', 'Sidoluhur': 'Jawa Tengah / Yogyakarta',
    'Sogan': 'Solo / Surakarta', 'Solo_Parang': 'Solo / Surakarta',
    'Sulawesi_Selatan_Lontara': 'Sulawesi Selatan',
    'Sumatera_Barat_Rumah_Minang': 'Sumatera Barat',
    'Sumatera_Utara_Boraspati': 'Sumatera Utara',
    'Tambal': 'Jawa', 'Yogyakarta_Kawung': 'Yogyakarta',
    'Yogyakarta_Parang': 'Yogyakarta',
}

WEAK_CLASSES = {'Priangan_Merak_Ngibing', 'Sogan', 'Lasem'}

# ── Page Title ──────────────────────────────────────────────────────────────
st.title("AI Deteksi Motif Batik Nusantara")
st.markdown("""
<div style="text-align:center; font-family:'Lato',sans-serif;
            font-size:0.95rem; color:#6B4226; margin: -8px 0 16px 0;
            font-style:italic;">
    Unggah foto kain batik — biarkan AI mengungkap nama, asal, dan filosofinya
</div>
<div style="text-align:center; margin-bottom:24px;">
    <svg width="320" height="14" viewBox="0 0 320 14" fill="none" xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="7" x2="140" y2="7" stroke="#E8C9A0" stroke-width="1"/>
        <circle cx="160" cy="7" r="5" fill="#C9962A"/>
        <circle cx="148" cy="7" r="2.5" fill="#E8C9A0"/>
        <circle cx="172" cy="7" r="2.5" fill="#E8C9A0"/>
        <line x1="180" y1="7" x2="320" y2="7" stroke="#E8C9A0" stroke-width="1"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# ── Columns ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([11, 9])

with col1:
    st.markdown("<p style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#C4824A; margin-bottom:6px;'>Input Gambar</p>", unsafe_allow_html=True)
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
    st.markdown("<p style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#C4824A; margin-bottom:6px;'>Hasil Analisis</p>", unsafe_allow_html=True)

    if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
        if model is None:
            st.error("Model gagal dimuat. Periksa file batik_model_v2.tflite.")
        else:
            with st.spinner('AI sedang menganalisis motif...'):
                processed_img = preprocess_image(image)
                input_details  = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], processed_img)
                model.invoke()
                predictions = model.get_tensor(output_details[0]['index'])

            predicted_class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions) * 100)
            raw_label    = class_names[predicted_class_idx]
            display_name = raw_label.replace("_", " ")
            description  = BATIK_INFO.get(raw_label, "Informasi motif ini belum tersedia.")
            region       = REGION_MAP.get(raw_label, "Indonesia")

            # Confidence bar color
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

            st.markdown(f"""
            <div class="batik-card">
                <span class="region-badge">Asal: {region}</span>
                <div class="motif-title">{display_name}</div>
                <p class="motif-desc">{description}</p>
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

            # Top-5 chart
            top5 = np.argsort(predictions[0])[-5:][::-1]
            top5_names = [class_names[i].replace("_", " ") for i in top5]
            top5_probs = predictions[0][top5]
            st.markdown("""
            <p style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.08em; color:#C4824A; margin: 20px 0 4px 0;">
                Kemungkinan Motif Lain
            </p>""", unsafe_allow_html=True)
            st.bar_chart(dict(zip(top5_names, top5_probs)), color="#8B4513")
            # Accessible text summary
            top5_txt = " · ".join([f"{n} ({p*100:.0f}%)" for n, p in zip(top5_names, top5_probs)])
            st.caption(f"Prediksi: {top5_txt}")

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