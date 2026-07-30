# Dashboard Streamlit – Analisis Performa Akademik Siswa

Dashboard ini dibuat untuk proyek analisis dataset **Student Alcohol Consumption**. Isinya meliputi ringkasan data, EDA, heatmap korelasi, evaluasi regresi linear, dan fitur prediksi nilai akhir G3.

## Cara menjalankan

1. Buka terminal pada folder `dashboard_streamlit`.
2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan dashboard:

   ```bash
   streamlit run app.py
   ```

4. Buka alamat yang muncul di terminal (umumnya `http://localhost:8501`).

## Data asli sudah disertakan

Tidak ada unggah CSV manual. ZIP ini sudah memuat dua file dataset asli pada folder `data`:

- `data/student-mat.csv` — 395 observasi
- `data/student-por.csv` — 649 observasi

Karena itu dashboard bisa langsung dijalankan setelah ZIP diunggah/di-*clone* dari GitHub, tanpa koneksi internet dan tanpa mengunggah file apa pun.

Jika suatu saat folder `data` tidak sengaja terhapus, aplikasi memiliki mekanisme cadangan untuk mengunduh dataset publik dari Kaggle, lalu dari sumber asli UCI Machine Learning Repository.

## Catatan metodologis

- Target model adalah **G3** (nilai akhir).
- Model menggunakan pembagian data **80:20** dan `random_state=42`.
- Fitur dipilih secara dinamis: 10 variabel numerik dengan korelasi absolut tertinggi terhadap G3.
- **G2 tidak digunakan** sebagai prediktor karena nilainya sangat dekat dengan G3 dan berisiko menimbulkan *data leakage*.
- Dashboard menampilkan metrik MAE, MSE, RMSE, dan R².
