# Sistem CBR Sengketa Tanah - Sistem Prediksi Putusan ⚖️

Sistem ini adalah sebuah **Dashboard Interaktif Berbasis Streamlit** yang dirancang untuk menganalisis dan memprediksi putusan perkara sengketa tanah menggunakan metode **Case-Based Reasoning (CBR)**. 

Dengan membandingkan fakta-fakta hukum dari kasus baru dengan kumpulan kasus historis terdahulu, sistem ini menghitung tingkat kemiripan menggunakan algoritma **TF-IDF** dan **Cosine Similarity** untuk menyarankan prediksi putusan hukum yang paling relevan.

---

## 🚀 Fitur Utama

1. **Prediksi Putusan Hukum (CBR Engine)**:
   - Pengguna dapat memasukkan ringkasan fakta kasus sengketa tanah yang baru.
   - Sistem melakukan pencarian kesamaan hukum secara *real-time* berbasis representasi teks.
   - Menampilkan daftar kasus historis yang paling mirip (Top-K) lengkap dengan persentase skor kemiripan (*similarity score*).
   - Memberikan rekomendasi putusan (misal: "Gugatan Dikabulkan", "Gugatan Ditolak", atau "Gugatan Tidak Dapat Diterima") berdasarkan voting berbobot kesamaan dari kasus terdahulu.

2. **Eksplorasi Basis Kasus**:
   - Memungkinkan pengguna mencari, memfilter, dan membaca detail kasus-kasus sengketa tanah historis yang tersimpan dalam basis data sistem.

3. **Visualisasi Statistik & Analisis**:
   - Grafik interaktif mengenai distribusi jenis sengketa, sebaran amar putusan, dan tren kasus tanah untuk membantu pemahaman data secara makro.

---

## 📁 Struktur Folder Utama

```text
├── app.py                  # Aplikasi utama Streamlit (UI & CBR Logic)
├── requirements.txt        # Daftar dependensi pustaka Python
├── data/
│   └── processed/
│       ├── cases.json      # Basis data kasus sengketa tanah (format JSON)
│       └── cases.csv       # Basis data kasus sengketa tanah (format CSV)
└── models/
    ├── svm_model.pkl       # Model Klasifikasi SVM (jika digunakan)
    └── tfidf_vectorizer.pkl# Ekstraktor Fitur TF-IDF (jika digunakan)
```

---

## 💻 Cara Menjalankan Aplikasi di Komputer Lokal

Ikuti langkah-langkah mudah berikut untuk menjalankan dashboard ini di komputer Anda secara lokal:

### 1. Prasyarat
Pastikan Anda sudah menginstal **Python** versi terbaru (disarankan versi **3.9 ke atas**) di sistem Anda. Anda dapat memeriksanya melalui terminal atau command prompt:
```bash
python --version
```

### 2. Unduh atau Ekstrak Proyek
Unduh kode sumber proyek ini (sebagai file ZIP) dan ekstrak ke folder pilihan Anda di komputer lokal.

### 3. Buka Terminal / Command Prompt
Buka aplikasi terminal (Linux/macOS) atau Command Prompt / PowerShell (Windows) lalu arahkan ke folder tempat Anda mengekstrak proyek ini:
```bash
cd /jalur/ke/folder/cbr-sengketa-tanah
```

### 4. Membuat Virtual Environment (Sangat Disarankan)
Guna menghindari konflik antar library di komputer Anda, buatlah lingkungan virtual khusus:
- **Di Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **Di macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 5. Instalasi Dependensi
Instal seluruh pustaka Python yang diperlukan dengan menjalankan perintah:
```bash
pip install -r requirements.txt
```

### 6. Menjalankan Dashboard Streamlit
Setelah instalasi selesai, jalankan perintah berikut untuk mengaktifkan server Streamlit:
```bash
streamlit run app.py
```

### 7. Mengakses Aplikasi
Setelah perintah dijalankan, Streamlit akan secara otomatis membuka jendela baru di browser default Anda. Jika tidak terbuka secara otomatis, Anda dapat menyalin dan mengunjungi alamat lokal berikut di browser Anda:
```text
http://localhost:8501
```

---

## 💡 Informasi Teknis Pendukung
- **Algoritma CBR**: Penghitungan TF-IDF (*Term Frequency - Inverse Document Frequency*) dan *Cosine Similarity* pada aplikasi ini diimplementasikan secara manual secara murni menggunakan pustaka bawaan Python (`math`, `re`, `json`). Hal ini membuat aplikasi sangat ringan, mandiri, dan bebas dari kendala kompatibilitas versi pustaka eksternal.
- **Data Sumber**: Aplikasi memuat data yang telah diproses dari berkas `data/processed/cases.json` saat pertama kali dijalankan dan menggunakan mekanisme `@st.cache_data` agar pemuatan ulang halaman berjalan instan.
