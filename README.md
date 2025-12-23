# Spam Detection (Random Forest + TF-IDF) — Streamlit App

Project ini adalah aplikasi sederhana untuk mengklasifikasikan teks menjadi **spam** atau **non-spam (ham)** menggunakan model **Random Forest**. Untuk mengubah teks menjadi fitur numerik, sistem memakai **TF-IDF vectorizer**. Model dan vectorizer sudah disimpan dalam file `.pkl`, lalu digunakan oleh aplikasi **Streamlit** untuk inference.

Repo ini memang ditujukan untuk dijalankan dari file `streamlit_spam_app.py`.

## Brief Penjelasan Project
Spam detection adalah salah satu kasus umum NLP (Natural Language Processing) untuk memfilter pesan yang tidak diinginkan. Di project ini, teks diubah menjadi representasi TF-IDF, lalu diklasifikasikan menggunakan Random Forest yang relatif stabil dan mudah dipakai untuk baseline klasifikasi.

## Project Overview
Yang dilakukan sistem secara garis besar:
1. Menerima input teks dari pengguna (melalui UI Streamlit).
2. Melakukan transformasi teks menjadi fitur TF-IDF menggunakan `tfidf_vectorizer.pkl`.
3. Memprediksi kelas menggunakan model `rf_model.pkl`.
4. Menampilkan hasil prediksi ke pengguna (spam / ham).

Komponen yang tersedia:
- Notebook untuk training/eksperimen.
- Artefak model hasil training (vectorizer + model).
- Aplikasi Streamlit untuk menjalankan prediksi secara interaktif.

## System Architecture
Arsitektur inference (saat aplikasi dijalankan):

User Input Text  
→ TF-IDF Vectorizer (`tfidf_vectorizer.pkl`)  
→ Random Forest Model (`rf_model.pkl`)  
→ Prediction Result (Spam / Ham)  
→ Streamlit UI

Arsitektur training (di notebook):
- Notebook melakukan preprocessing + training model Random Forest.
- Notebook menyimpan:
  - `tfidf_vectorizer.pkl` (TF-IDF transformer)
  - `rf_model.pkl` (model Random Forest terlatih)

Catatan: Aplikasi Streamlit tidak melatih model dari awal. Ia hanya memuat artefak `.pkl` untuk inference.

## Project Structure
Struktur file utama di repo ini:

.
├── README.md  
├── nlt_base_code.ipynb  
├── rf_model.pkl  
├── streamlit_spam_app.py  
└── tfidf_vectorizer.pkl  

Penjelasan singkat:
- `streamlit_spam_app.py`  
  Entry point aplikasi Streamlit untuk prediksi spam/ham.
- `tfidf_vectorizer.pkl`  
  TF-IDF vectorizer yang dipakai untuk mengubah teks menjadi fitur.
- `rf_model.pkl`  
  Model Random Forest terlatih untuk klasifikasi.
- `nlt_base_code.ipynb`  
  Notebook untuk proses training/eksperimen (dan biasanya tempat menghasilkan file `.pkl`).

## Cara Run Locally

### 1) Prerequisites
- Python 3.9+ (disarankan)
- pip

### 2) Clone repository
```bash
git clone https://github.com/freyzaza/spam-detection-random-forest.git
cd spam-detection-random-forest
```

### 3) Buat virtual environment (opsional tapi disarankan)
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4) Install dependencies
Karena repo belum menyediakan `requirements.txt`, install paket minimal berikut:
```bash
pip install streamlit scikit-learn numpy
```

Jika di kode ada penggunaan `pandas` atau `nltk`, tambahkan:
```bash
pip install pandas nltk
```

### 5) Jalankan aplikasi Streamlit
```bash
streamlit run streamlit_spam_app.py
```

Setelah itu buka URL yang muncul di terminal (biasanya `http://localhost:8501`).

## (Opsional) Retrain Model
Jika kamu ingin melatih ulang model:
1. Buka `nlt_base_code.ipynb`.
2. Jalankan cell dari awal sampai akhir.
3. Pastikan output training menyimpan ulang:
   - `tfidf_vectorizer.pkl`
   - `rf_model.pkl`
4. Jalankan lagi aplikasi Streamlit seperti biasa.
