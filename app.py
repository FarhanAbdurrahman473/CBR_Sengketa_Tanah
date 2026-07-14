import streamlit as st
import pandas as pd
import json
import re
import math
from pathlib import Path

# Set Page Config
st.set_page_config(
    page_title="Sistem CBR Sengketa Tanah",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DATA AND CBR ENGINE DEFINITION ---

def tokenize(text):
    if not text:
        return []
    # Lowercase, remove non-alphanumeric, and filter short words
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    words = cleaned.split()
    return [w for w in words if len(w) > 2]

class TfidfEngine:
    def __init__(self, cases):
        self.cases = cases
        self.docs = []
        for c in cases:
            # Build a rich text representation of each case for TF-IDFindexing
            text = f"{c.get('ringkasan_fakta', '')} {c.get('penggugat', '')} {c.get('tergugat', '')} {c.get('objek_sengketa', '')}"
            self.docs.append(tokenize(text))
        
        self.doc_frequencies = {}
        self.idfs = {}
        self.doc_vectors = []
        
        num_docs = len(self.docs)
        for doc in self.docs:
            unique_words = set(doc)
            for word in unique_words:
                self.doc_frequencies[word] = self.doc_frequencies.get(word, 0) + 1
                
        for word, freq in self.doc_frequencies.items():
            self.idfs[word] = math.log(1 + num_docs / freq)
            
        for doc in self.docs:
            self.doc_vectors.append(self.create_vector(doc))
            
    def create_vector(self, doc):
        term_counts = {}
        for word in doc:
            term_counts[word] = term_counts.get(word, 0) + 1
            
        vector = {}
        total_terms = len(doc) if len(doc) > 0 else 1
        for word, count in term_counts.items():
            tf = count / total_terms
            idf = self.idfs.get(word, 0)
            vector[word] = tf * idf
        return vector

    def query(self, query_text, top_k=5):
        query_tokens = tokenize(query_text)
        query_vec = self.create_vector(query_tokens)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = cosine_similarity(query_vec, doc_vec)
            scores.append((i, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def cosine_similarity(vec_a, vec_b):
    dot_product = 0.0
    mag_a = 0.0
    mag_b = 0.0
    
    for val in vec_a.values():
        mag_a += val * val
    for val in vec_b.values():
        mag_b += val * val
        
    for word, val_a in vec_a.items():
        val_b = vec_b.get(word, 0.0)
        dot_product += val_a * val_b
        
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (math.sqrt(mag_a) * math.sqrt(mag_b))

def predict_case(query, cases, engine):
    if not query.strip():
        return "Lainnya", []
        
    top_results = engine.query(query, 5)
    retrieved_cases = []
    for idx, score in top_results:
        c = cases[idx].copy()
        c['similarity'] = score
        retrieved_cases.append(c)
        
    if not retrieved_cases:
        return "Lainnya", []
        
    # Voting among Top 5
    label_counts = {}
    for c in retrieved_cases:
        lbl = c.get('label', 'Lainnya')
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
    best_label = ''
    max_count = -1
    highest_sim_for_label = -1
    
    for lbl, count in label_counts.items():
        highest_sim = max([c['similarity'] for c in retrieved_cases if c.get('label') == lbl], default=0.0)
        if count > max_count or (count == max_count and highest_sim > highest_sim_for_label):
            max_count = count
            best_label = lbl
            highest_sim_for_label = highest_sim
            
    return best_label, retrieved_cases

# Load the case dataset
@st.cache_data
def load_cases():
    file_path = Path("data/processed/cases.json")
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

cases = load_cases()
tfidf_engine = TfidfEngine(cases) if cases else None

# --- 2. CUSTOM THEME AND STYLING (CSS Injection) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background-color: #064e3b;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    
    /* Metrics panel */
    .kpi-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Custom verdict badges */
    .verdict-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 1.1rem;
        letter-spacing: 0.025em;
        margin-bottom: 1rem;
    }
    .verdict-dikabulkan {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .verdict-sebagian {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .verdict-ditolak {
        background-color: #fff5f5;
        color: #9b2c2c;
        border: 1px solid #feb2b2;
    }
    .verdict-tidak-diterima {
        background-color: #f8fafc;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    
    /* Info container */
    .case-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HEADER SECTION ---
st.markdown("""
    <div class="main-header">
        <h1>⚖️ Sistem CBR Sengketa Tanah</h1>
        <p>Dashboard Case-Based Reasoning (CBR) untuk Analisis dan Prediksi Putusan Sengketa Tanah • Universitas Pasir Pengaraian</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Putusan", "📂 Precedent Directory", "📊 Analisis & Metrik"])

# Quick Examples List
EXAMPLES = [
    {
        "title": "Kasus 1: Balik Nama SHM Tergugat Hilang",
        "description": "Penggugat membeli sebidang tanah pertanian Sertifikat Hak Milik (SHM) No. 01439 dari Tergugat pada tahun 2000. Jual beli dibuktikan dengan Surat Keterangan Kepala Desa. Namun Tergugat sekarang tidak diketahui lagi keberadaannya di seluruh wilayah RI, sehingga sertifikat belum dapat dibalik nama."
    },
    {
        "title": "Kasus 2: Penyerobotan / Pendudukan Tanpa Hak",
        "description": "Penggugat adalah pemilik sah sebidang tanah berdasarkan Sertifikat Hak Milik. Tergugat menguasai tanah tersebut tanpa hak, mendirikan bangunan permanen di atasnya, serta menolak mengosongkan lahan meskipun telah ditegur berulang kali secara melawan hukum."
    },
    {
        "title": "Kasus 3: Sengketa Waris dan Double Sertifikat",
        "description": "Sengketa kepemilikan sebidang tanah warisan yang di atasnya terbit sertifikat ganda. Tergugat mengklaim kepemilikan tanah dengan sertifikat baru yang tumpang tindih dengan Sertifikat Hak Milik asli yang sudah terdaftar sejak tahun 1991."
    }
]

# --- TAB 1: PREDICTION ENGINE ---
with tab1:
    col_left, col_right = st.columns([5, 7])
    
    with col_left:
        st.subheader("Kronologi Sengketa Baru")
        st.caption("Ketik atau pilih cerita kasus baru untuk diprediksi hasilnya.")
        
        # Helper to apply example
        example_choice = st.selectbox(
            "Pilih contoh kasus untuk mengisi cepat:",
            ["-- Silakan Pilih Contoh Kasus --"] + [ex["title"] for ex in EXAMPLES]
        )
        
        initial_text = ""
        if example_choice != "-- Silakan Pilih Contoh Kasus --":
            selected_ex = next(ex for ex in EXAMPLES if ex["title"] == example_choice)
            initial_text = selected_ex["description"]
            
        query_text = st.text_area(
            "Kronologi Lengkap Perkara:",
            value=initial_text,
            height=280,
            placeholder="Masukkan deskripsi perkara disini..."
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            clear_btn = st.button("Bersihkan", use_container_width=True)
            if clear_btn:
                st.rerun()
        with col_btn2:
            predict_btn = st.button("Prediksi Putusan", type="primary", use_container_width=True)
            
    with col_right:
        if predict_btn and query_text.strip():
            with st.spinner("Melakukan kalkulasi TF-IDF & Cosine Similarity..."):
                pred_lbl, retrieved = predict_case(query_text, cases, tfidf_engine)
                
                st.subheader("Hasil Analisis Prediksi")
                
                # Dynamic badge styles
                badge_class = "verdict-tidak-diterima"
                if pred_lbl == "Dikabulkan":
                    badge_class = "verdict-dikabulkan"
                elif pred_lbl == "Dikabulkan Sebagian":
                    badge_class = "verdict-sebagian"
                elif pred_lbl == "Ditolak":
                    badge_class = "verdict-ditolak"
                
                st.markdown(f"""
                    <div class="verdict-badge {badge_class}">
                        Prediksi Putusan: {pred_lbl}
                    </div>
                """, unsafe_allow_html=True)
                
                st.info("Sistem menganalisis 5 yurisprudensi terdekat dengan kesamaan tertinggi dan menggunakan voting mayoritas (K-Nearest Neighbors).")
                
                # Horizontal Similarity chart
                sim_data = pd.DataFrame({
                    'Nomor Perkara': [r['nomor_perkara'] for r in retrieved],
                    'Match %': [r['similarity'] * 100 for r in retrieved],
                    'Label': [r['label'] for r in retrieved]
                })
                
                st.markdown("##### Nilai Kemiripan Kasus Terdekat (%)")
                st.bar_chart(sim_data.set_index('Nomor Perkara')['Match %'])
                
                # Display Expanders for Matched Cases
                st.markdown("##### Detail Yurisprudensi Pendukung (Top 5)")
                for idx, c in enumerate(retrieved):
                    lbl_badge = "🟢" if c['label'] == 'Dikabulkan' else ("🟡" if c['label'] == 'Dikabulkan Sebagian' else "🔴")
                    with st.expander(f"#{idx+1} | {c['nomor_perkara']} ({c['similarity']*100:.2f}% Match) - {lbl_badge} {c['label']}"):
                        st.markdown(f"**Pengadilan:** {c.get('pengadilan', 'PN Pasir Pengaraian')}")
                        st.markdown(f"**Para Pihak:** {c.get('penggugat', '')} **Lawan** {c.get('tergugat', '')}")
                        st.markdown(f"**Objek Sengketa:** {c.get('objek_sengketa', '-')} • **Luas Lahan:** {c.get('luas_tanah', '-')} M²")
                        st.markdown(f"**Ringkasan Fakta Hukum:**")
                        st.write(c.get('ringkasan_fakta', ''))
                        if c.get('amar_putusan'):
                            st.markdown(f"**Amar Putusan:**")
                            st.info(c.get('amar_putusan'))
        else:
            st.markdown("""
                <div style="text-align: center; padding: 5rem 2rem; color: #64748b; background-color: #f8fafc; border: 2px dashed #e2e8f0; border-radius: 12px; margin-top: 1.5rem;">
                    <span style="font-size: 3rem;">🔮</span>
                    <h4 style="margin: 1rem 0 0.5rem 0; font-weight: 700; color: #334155;">Menunggu Input Perkara</h4>
                    <p style="font-size: 0.85rem; margin: 0;">Silakan masukkan detail sengketa baru di panel kiri, lalu tekan tombol <b>Prediksi Putusan</b>.</p>
                </div>
            """, unsafe_allow_html=True)


# --- TAB 2: PRECEDENT DIRECTORY ---
with tab2:
    st.subheader("Daftar Dokumen Yurisprudensi Sengketa")
    st.caption("Telusuri dan cari berkas putusan sengketa tanah yang tersimpan di dalam basis kasus.")
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_query = st.text_input("🔍 Cari Kasus (berdasarkan nomor, para pihak, atau ringkasan fakta):", "")
    with col_f2:
        label_filter = st.selectbox(
            "Filter Berdasarkan Putusan:",
            ["Semua", "Dikabulkan", "Dikabulkan Sebagian", "Ditolak", "Tidak Diterima"]
        )
        
    filtered_cases = cases
    if search_query:
        filtered_cases = [c for c in filtered_cases if (
            search_query.lower() in c['nomor_perkara'].lower() or
            search_query.lower() in c['penggugat'].lower() or
            search_query.lower() in c['tergugat'].lower() or
            search_query.lower() in c['ringkasan_fakta'].lower()
        )]
    if label_filter != "Semua":
        filtered_cases = [c for c in filtered_cases if c['label'] == label_filter]
        
    st.markdown(f"Menampilkan **{len(filtered_cases)}** dari **{len(cases)}** putusan.")
    
    # Render table/cards
    for c in filtered_cases:
        lbl_style = "🟢 Dikabulkan" if c['label'] == 'Dikabulkan' else ("🟡 Dikabulkan Sebagian" if c['label'] == 'Dikabulkan Sebagian' else "🔴 Ditolak")
        if c['label'] == 'Tidak Diterima':
            lbl_style = "⚪ Tidak Diterima"
            
        with st.container():
            st.markdown(f"""
                <div class="case-container">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <span style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">ID #{c['case_id']} • {c.get('pengadilan', 'PN Pasir Pengaraian')}</span>
                            <h4 style="margin: 0.1rem 0 0.5rem 0; font-weight: 800; color: #0f172a;">{c['nomor_perkara']}</h4>
                        </div>
                        <span style="font-weight: 700; font-size: 0.8rem; background-color: #f1f5f9; padding: 0.25rem 0.6rem; border-radius: 6px; color: #334155;">{lbl_style}</span>
                    </div>
                    <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">{c['ringkasan_fakta']}</p>
                    <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: #64748b;">
                        <span>👤 <b>Penggugat:</b> {c['penggugat']}</span>
                        <span>👥 <b>Tergugat:</b> {c['tergugat']}</span>
                        <span>🗺️ <b>Objek:</b> {c.get('objek_sengketa', 'SHM')} ({c.get('luas_tanah', '-')} M²)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# --- TAB 3: ANALISIS & METRIK ---
with tab3:
    st.subheader("Statistik & Metrik Basis Kasus")
    
    # KPI Grid
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-val">40</div>
                <div class="kpi-lbl">Total Basis Perkara</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-val">82.5%</div>
                <div class="kpi-lbl">Rasio Dikabulkan</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-val">Pasir Pengaraian</div>
                <div class="kpi-lbl">Wilayah Hukum</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-val">7.725 m²</div>
                <div class="kpi-lbl">Rata-rata Luas Sengketa</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graphs
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### Distribusi Putusan Sengketa")
        if cases:
            labels = [c['label'] for c in cases]
            label_counts = pd.Series(labels).value_counts().reset_index()
            label_counts.columns = ['Hasil Putusan', 'Jumlah Kasus']
            st.bar_chart(label_counts.set_index('Hasil Putusan')['Jumlah Kasus'])
            
    with col_chart2:
        st.markdown("##### Sebaran Pihak Penggugat Terbanyak")
        if cases:
            # Quick parsing of most common plaintiff keywords
            penggugat_list = [c['penggugat'] for c in cases]
            p_counts = pd.Series(penggugat_list).value_counts().head(5).reset_index()
            p_counts.columns = ['Nama Penggugat', 'Jumlah Perkara']
            st.dataframe(p_counts, use_container_width=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Theory of CBR
    st.markdown("### Prinsip Siklus Kerja Case-Based Reasoning (CBR)")
    col_cbr1, col_cbr2, col_cbr3, col_cbr4 = st.columns(4)
    
    with col_cbr1:
        st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1.25rem; min-height: 180px;">
                <span style="font-size: 1.5rem; font-weight: 800; color: #166534;">1. RETRIEVE</span>
                <p style="font-size: 0.8rem; color: #166534; font-weight: bold; margin: 0.5rem 0 0.25rem 0;">Temukan Kembali</p>
                <p style="font-size: 0.75rem; color: #4b5563; margin: 0;">Mencari kasus terdahulu yang paling mirip dengan kronologi sengketa baru menggunakan TF-IDF dan Cosine Similarity.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_cbr2:
        st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1.25rem; min-height: 180px;">
                <span style="font-size: 1.5rem; font-weight: 800; color: #166534;">2. REUSE</span>
                <p style="font-size: 0.8rem; color: #166534; font-weight: bold; margin: 0.5rem 0 0.25rem 0;">Gunakan Kembali</p>
                <p style="font-size: 0.75rem; color: #4b5563; margin: 0;">Menerapkan keputusan yurisprudensi dari kasus-kasus lama yang serupa untuk menyarankan label putusan sengketa baru.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_cbr3:
        st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1.25rem; min-height: 180px;">
                <span style="font-size: 1.5rem; font-weight: 800; color: #166534;">3. REVISE</span>
                <p style="font-size: 0.8rem; color: #166534; font-weight: bold; margin: 0.5rem 0 0.25rem 0;">Revisi Hasil</p>
                <p style="font-size: 0.75rem; color: #4b5563; margin: 0;">Pakar hukum meninjau kesesuaian rekomendasi dan menyesuaikannya dengan kondisi riil atau regulasi terkini.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_cbr4:
        st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1.25rem; min-height: 180px;">
                <span style="font-size: 1.5rem; font-weight: 800; color: #166534;">4. RETAIN</span>
                <p style="font-size: 0.8rem; color: #166534; font-weight: bold; margin: 0.5rem 0 0.25rem 0;">Simpan Perkara</p>
                <p style="font-size: 0.75rem; color: #4b5563; margin: 0;">Menyimpan kasus baru yang sudah diputus secara final ke dalam pangkalan basis pengetahuan yurisprudensi.</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <br><br>
    <div style="text-align: center; color: #94a3b8; font-size: 0.75rem; padding: 1.5rem 0; border-top: 1px solid #e2e8f0;">
        © 2026 Sistem CBR Sengketa Tanah • Universitas Pasir Pengaraian.
    </div>
""", unsafe_allow_html=True)
