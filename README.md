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

## Memakai data asli

Dashboard dapat langsung dibuka dalam **mode demonstrasi** agar tampilannya dapat diperiksa. Namun, hasilnya hanya bersifat contoh karena memakai data sintetis.

Untuk memakai data penelitian yang sebenarnya, unggah kedua file berikut melalui sidebar dashboard:

- `student-mat.csv`
- `student-por.csv`

Alternatifnya, buat folder `data` di dalam folder dashboard lalu simpan kedua CSV tersebut di sana. Nama file harus sama persis.

Dataset asli dapat diunduh dari Kaggle: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption

## Catatan metodologis

- Target model adalah **G3** (nilai akhir).
- Model menggunakan pembagian data **80:20** dan `random_state=42`.
- Fitur dipilih secara dinamis: 10 variabel numerik dengan korelasi absolut tertinggi terhadap G3.
- **G2 tidak digunakan** sebagai prediktor karena nilainya sangat dekat dengan G3 dan berisiko menimbulkan *data leakage*.
- Dashboard menampilkan metrik MAE, MSE, RMSE, dan R².
