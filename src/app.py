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
    st.title("ℹ️ Tentang Projek")
    st.markdown("---")
    st.write("""
    **Aplikasi Klasifikasi Batik** ini dikembangkan untuk melestarikan dan memperkenalkan kekayaan motif Nusantara melalui kecerdasan buatan.
    """)
    
    st.subheader("🛠️ Teknologi")
    st.code("Python 3.10\nStreamlit\nTensorFlow Lite\nEfficientNetV2S")
    
    st.subheader("📊 Performa Model")
    st.write("TTA Accuracy: **84.68%**")
    st.write("Macro F1: **81.45%**")
    
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
        st.image(image, caption='Preview Gambar', use_container_width=True)
        
        predict_btn = st.button('🔍 IDENTIFIKASI MOTIF', use_container_width=True)

with col2:
    st.subheader("📝 Hasil Analisis")
    
    # Weak classes with F1 < 0.65 — model may be less reliable for these
    WEAK_CLASSES = {'Priangan_Merak_Ngibing', 'Sogan', 'Lasem'}

    if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
        if model is None:
            st.error("Model gagal dimuat. Periksa file model batik_model_v2.tflite.")
        else:
            with st.spinner('Memproses gambar...'):
                processed_img = preprocess_image(image)

                input_details  = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], processed_img)
                model.invoke()
                predictions = model.get_tensor(output_details[0]['index'])

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

                if confidence < 60:
                    st.warning("⚠️ Tingkat keyakinan rendah (< 60%). Hasil prediksi mungkin kurang akurat — coba gunakan foto yang lebih jelas.")
                elif raw_label in WEAK_CLASSES:
                    st.warning(f"⚠️ Motif **{display_name}** termasuk kelas dengan akurasi lebih rendah (data pelatihan terbatas). Interpretasikan hasil ini dengan hati-hati.")

                st.write("")
                st.caption("Kemungkinan motif lain:")
                top_5_idx = np.argsort(predictions[0])[-5:][::-1]
                top_5_probs = predictions[0][top_5_idx]
                top_5_names = [class_names[i].replace("_", " ") for i in top_5_idx]
                st.bar_chart(dict(zip(top_5_names, top_5_probs)), color="#8B4513")

    elif uploaded_file is None:
        st.info("Unggah gambar di panel kiri untuk memulai analisis.")