import streamlit as st
from utils.data import REGION_MAP

def render_sidebar():
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
        with st.expander("Cara Penggunaan", expanded=True):
            st.markdown("""
            <div style="font-size:0.85rem; line-height:1.7; color:#F5EDD4;">
                <b style="color:#C9962A;">1.</b> Unggah foto kain batik atau gunakan kamera<br>
                <b style="color:#C9962A;">2.</b> Klik tombol <em>Identifikasi Motif</em><br>
                <b style="color:#C9962A;">3.</b> Temukan nama, asal, dan filosofi motif
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
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
        with st.expander("28 Motif Batik"):
            island_groups = {}
            for motif, island in REGION_MAP.items():
                display_m = motif.replace("_", " ")
                if island not in island_groups:
                    island_groups[island] = []
                island_groups[island].append(display_m)
            
            for island in sorted(island_groups.keys()):
                st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#E8C9A0; margin-top:10px;'>{island.upper()}</div>", unsafe_allow_html=True)
                for m in sorted(island_groups[island]):
                    st.markdown(f"<div style='font-size:0.8rem; color:#F5EDD4; margin-left:8px; line-height:1.5;'>• {m}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.expander("Tentang Proyek"):
            st.markdown("""
            <div style="font-size:0.83rem; line-height:1.6; color:#F5EDD4;">
                Aplikasi ini dikembangkan untuk melestarikan dan memperkenalkan
                kekayaan motif batik Nusantara melalui kecerdasan buatan berbasis
                <em>EfficientNetV2S</em> dengan 5-Fold Cross-Validation.
            </div>
            """, unsafe_allow_html=True)
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
