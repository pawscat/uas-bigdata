"""Dashboard Streamlit – Analisis Performa Akademik Siswa.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Dashboard Performa Akademik", page_icon="📚", layout="wide"
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background:#ffffff; border:1px solid #e5e7eb;
        padding:15px; border-radius:14px; box-shadow:0 2px 8px rgba(15,23,42,.05);}
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"] {color:#0f172a !important;}
      .small-note {color:#64748b; font-size:.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
SUBJECT_COL = "Mata Pelajaran"
SUBJECTS = {"Matematika": "student-mat.csv", "Bahasa Portugis": "student-por.csv"}
DATA_SOURCES = [
    ("Kaggle", "https://www.kaggle.com/api/v1/datasets/download/uciml/student-alcohol-consumption"),
    ("UCI Machine Learning Repository", "https://archive.ics.uci.edu/static/public/320/student+performance.zip"),
]

def read_dataset(local_path: Path) -> pd.DataFrame:
    """Membaca CSV UCI/Kaggle dengan pemisah titik koma atau koma secara otomatis."""
    return pd.read_csv(local_path, sep=None, engine="python")


def download_dataset() -> str:
    """Unduh sekali dari Kaggle publik; UCI dipakai sebagai cadangan bila diperlukan."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for source_name, url in DATA_SOURCES:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=35) as response:
                archive_bytes = response.read()
            with ZipFile(BytesIO(archive_bytes)) as archive:
                members = {Path(name).name: name for name in archive.namelist()}
                if not all(filename in members for filename in SUBJECTS.values()):
                    raise ValueError("Arsip tidak memuat student-mat.csv dan student-por.csv.")
                for filename in SUBJECTS.values():
                    (DATA_DIR / filename).write_bytes(archive.read(members[filename]))
            return source_name
        except (URLError, TimeoutError, BadZipFile, OSError, ValueError) as exc:
            errors.append(f"{source_name}: {exc}")
    raise RuntimeError("Dataset tidak dapat diunduh otomatis. " + " | ".join(errors))


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
    """Gunakan cache lokal; jika belum ada, unduh otomatis lalu simpan ke folder data."""
    local_files_exist = all((DATA_DIR / filename).exists() for filename in SUBJECTS.values())
    source = "Cache lokal" if local_files_exist else f"Unduhan otomatis dari {download_dataset()}"
    mat = read_dataset(DATA_DIR / SUBJECTS["Matematika"])
    por = read_dataset(DATA_DIR / SUBJECTS["Bahasa Portugis"])
    mat[SUBJECT_COL] = "Matematika"
    por[SUBJECT_COL] = "Bahasa Portugis"
    data = pd.concat([mat, por], ignore_index=True, sort=False)
    if "G3" not in data.columns:
        raise ValueError("Kolom G3 tidak ditemukan pada dataset.")
    return data, source


def subject_label(value: object) -> str:
    return "≥ 10 (Lulus)" if value >= 10 else "< 10 (Belum lulus)"


def build_model(data: pd.DataFrame) -> dict | None:
    """Mereplikasi pendekatan notebook: Top 10 fitur numerik, tanpa G2 (leakage)."""
    numeric = data.select_dtypes(include=np.number).dropna(axis=0).copy()
    if "G3" not in numeric or len(numeric) < 20:
        return None
    correlations = numeric.corr(numeric_only=True)["G3"].abs().sort_values(ascending=False)
    features = [col for col in correlations.index if col not in {"G3", "G2"}][:10]
    if not features:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        numeric[features], numeric["G3"], test_size=0.20, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    prediction = model.predict(x_test)
    mse = mean_squared_error(y_test, prediction)
    return {
        "model": model,
        "features": features,
        "metrics": {
            "MAE": mean_absolute_error(y_test, prediction),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(y_test, prediction),
        },
        "coefficients": pd.DataFrame({"Fitur": features, "Koefisien": model.coef_}).sort_values("Koefisien"),
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(prediction).reset_index(drop=True),
        "feature_medians": numeric[features].median(),
        "feature_min": numeric[features].min(),
        "feature_max": numeric[features].max(),
    }


def grade_chart(data: pd.DataFrame):
    count = data.assign(Status=data["G3"].apply(subject_label)).groupby([SUBJECT_COL, "Status"], as_index=False).size()
    return px.bar(
        count, x=SUBJECT_COL, y="size", color="Status", barmode="group", text_auto=True,
        color_discrete_map={"≥ 10 (Lulus)": "#16a34a", "< 10 (Belum lulus)": "#ef4444"},
        labels={"size": "Jumlah siswa"}, title="Status kelulusan berdasarkan nilai akhir (G3)",
    )


def render_overview(data: pd.DataFrame, source: str):
    st.caption(f"Sumber saat ini: **{source}**")
    total = len(data)
    avg_g3 = data["G3"].mean()
    pass_rate = (data["G3"] >= 10).mean() * 100
    avg_failures = data["failures"].mean() if "failures" in data else np.nan
    a, b, c, d = st.columns(4)
    a.metric("Total observasi", f"{total:,}")
    b.metric("Rata-rata nilai akhir (G3)", f"{avg_g3:.2f} / 20")
    c.metric("Persentase lulus", f"{pass_rate:.1f}%")
    d.metric("Rata-rata kegagalan", f"{avg_failures:.2f}" if pd.notna(avg_failures) else "–")

    left, right = st.columns((1.15, 1))
    with left:
        st.plotly_chart(grade_chart(data), use_container_width=True)
    with right:
        avg_subject = data.groupby(SUBJECT_COL, as_index=False)[["G1", "G3"]].mean().melt(
            id_vars=SUBJECT_COL, var_name="Tahap nilai", value_name="Rata-rata nilai"
        )
        fig = px.bar(
            avg_subject, x=SUBJECT_COL, y="Rata-rata nilai", color="Tahap nilai", barmode="group",
            text_auto=".2f", color_discrete_map={"G1": "#60a5fa", "G3": "#1d4ed8"},
            title="Perbandingan rata-rata nilai awal (G1) dan akhir (G3)",
        )
        fig.update_yaxes(range=[0, 20])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ringkasan interpretasi")
    st.write(
        "Nilai G3 adalah nilai akhir siswa pada skala 0–20. Dashboard ini memisahkan "
        "Matematika dan Bahasa Portugis agar perbedaan pola akademik dapat diamati secara jelas. "
        "Untuk prediksi, G2 tidak digunakan karena terlalu dekat dengan nilai akhir dan dapat menyebabkan kebocoran informasi (data leakage)."
    )


def render_eda(data: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            data, x="G1", y="G3", color=SUBJECT_COL, trendline="ols",
            hover_data=["failures", "studytime", "Dalc"],
            title="Hubungan nilai periode pertama (G1) dan nilai akhir (G3)",
            labels={"G1": "Nilai periode pertama", "G3": "Nilai akhir"},
        )
        fig.update_xaxes(range=[0, 20]); fig.update_yaxes(range=[0, 20])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(
            data, x="failures", y="G3", color=SUBJECT_COL, points="outliers",
            title="Distribusi nilai akhir menurut jumlah kegagalan", labels={"failures": "Jumlah kegagalan"},
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        alcohol = data.groupby([SUBJECT_COL, "Dalc"], as_index=False)["G3"].mean()
        fig = px.line(
            alcohol, x="Dalc", y="G3", color=SUBJECT_COL, markers=True,
            title="Rata-rata nilai akhir menurut konsumsi alkohol harian (Dalc)",
            labels={"Dalc": "Konsumsi alkohol harian (1 rendah – 5 tinggi)", "G3": "Rata-rata G3"},
        )
        fig.update_yaxes(range=[0, 20])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        numeric = data.select_dtypes(include=np.number)
        corr = numeric.corr(numeric_only=True)["G3"].drop("G3").sort_values(key=np.abs, ascending=False).head(12)
        fig = px.bar(
            corr.reset_index(), x="G3", y="index", orientation="h", text_auto=".2f",
            color="G3", color_continuous_scale="RdBu", range_color=[-1, 1],
            title="12 korelasi terkuat dengan nilai akhir (G3)", labels={"index": "Variabel", "G3": "Korelasi"},
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap korelasi variabel numerik")
    selected = [c for c in ["G1", "G2", "G3", "failures", "studytime", "Medu", "Fedu", "Dalc", "Walc", "absences", "goout", "age"] if c in data]
    corr_matrix = data[selected].corr(numeric_only=True)
    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Korelasi variabel numerik utama")
    st.plotly_chart(fig, use_container_width=True)


def render_model(data: pd.DataFrame):
    available_subjects = data[SUBJECT_COL].dropna().unique().tolist()
    model_subject = st.selectbox("Dataset untuk pemodelan", available_subjects, key="model_subject")
    subset = data[data[SUBJECT_COL] == model_subject].copy()
    result = build_model(subset)
    if result is None:
        st.error("Data tidak cukup untuk membuat model regresi.")
        return

    st.caption("Model: regresi linear berganda, pembagian data 80:20, random_state=42, Top 10 fitur numerik berdasarkan korelasi absolut. Kolom G2 dikeluarkan dari fitur prediksi.")
    metric_cols = st.columns(4)
    for col, (name, value) in zip(metric_cols, result["metrics"].items()):
        col.metric(name, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        comparison = pd.DataFrame({"Aktual": result["actual"], "Prediksi": result["predicted"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison.index, y=comparison["Aktual"], mode="markers", name="Nilai aktual", marker={"color": "#1d4ed8"}))
        fig.add_trace(go.Scatter(x=comparison.index, y=comparison["Prediksi"], mode="markers", name="Nilai prediksi", marker={"color": "#f59e0b"}))
        fig.update_layout(title="Perbandingan nilai aktual dan prediksi pada data uji", xaxis_title="Observasi data uji", yaxis_title="G3")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        coef = result["coefficients"]
        fig = px.bar(coef, x="Koefisien", y="Fitur", orientation="h", text_auto=".3f", color="Koefisien", color_continuous_scale="RdBu", title="Koefisien model")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediksi nilai akhir siswa")
    st.caption("Masukkan kondisi siswa. Nilai awal pada setiap kolom mengikuti median dataset yang sedang dipilih.")
    inputs = {}
    input_columns = st.columns(2)
    for idx, feature in enumerate(result["features"]):
        median = float(result["feature_medians"][feature])
        lower = float(result["feature_min"][feature])
        upper = float(result["feature_max"][feature])
        step = 1.0 if lower.is_integer() and upper.is_integer() else 0.1
        with input_columns[idx % 2]:
            inputs[feature] = st.number_input(feature, min_value=lower, max_value=upper, value=median, step=step)
    if st.button("Hitung prediksi nilai G3", type="primary"):
        prediction = float(result["model"].predict(pd.DataFrame([inputs])[result["features"]])[0])
        prediction = np.clip(prediction, 0, 20)
        status = "diprediksi lulus" if prediction >= 10 else "diprediksi belum lulus"
        st.success(f"Prediksi nilai akhir (G3): **{prediction:.2f} / 20** — siswa {status}.")
    st.markdown("<p class='small-note'>Hasil ini adalah estimasi model untuk pembelajaran, bukan penilaian resmi siswa.</p>", unsafe_allow_html=True)


def render_data(data: pd.DataFrame):
    st.write(f"**Ukuran data:** {data.shape[0]:,} baris × {data.shape[1]} kolom")
    left, right = st.columns(2)
    with left:
        st.write("**Tipe data**")
        st.dataframe(pd.DataFrame({"Kolom": data.columns, "Tipe data": data.dtypes.astype(str)}), hide_index=True, use_container_width=True)
    with right:
        st.write("**Missing value**")
        missing = data.isna().sum().reset_index()
        missing.columns = ["Kolom", "Jumlah missing"]
        st.dataframe(missing, hide_index=True, use_container_width=True)
    st.write("**Pratinjau data**")
    st.dataframe(data, use_container_width=True, height=380)
    st.download_button("Unduh data yang sedang ditampilkan (CSV)", data.to_csv(index=False).encode("utf-8"), "data_dashboard.csv", "text/csv")


def main():
    st.title("📚 Dashboard Analisis Performa Akademik Siswa")
    st.write("Eksplorasi data, evaluasi model regresi, dan prediksi nilai akhir menggunakan dataset *Student Alcohol Consumption*.")
    with st.sidebar:
        st.header("Status data")
    try:
        with st.spinner("Menyiapkan dataset asli..."):
            data, source = load_data()
    except (RuntimeError, ValueError) as exc:
        st.error(f"Dashboard tidak dapat mengambil dataset: {exc}")
        st.info("Pastikan perangkat terhubung ke internet, lalu muat ulang halaman. Setelah berhasil sekali, data akan dibaca dari cache lokal.")
        st.stop()
    with st.sidebar:
        st.success("Dataset asli siap digunakan")
        st.caption("Data diambil otomatis sekali dari Kaggle publik dan disimpan ke cache lokal. Jalankan ulang berikutnya tidak perlu mengunduh maupun mengunggah file.")
        st.divider()
        st.header("Filter")
        subject = st.selectbox("Mata pelajaran", ["Semua"] + data[SUBJECT_COL].dropna().unique().tolist())
        grade_range = st.slider("Rentang nilai akhir (G3)", 0, 20, (0, 20))
        st.divider()
        st.caption("Sumber asli: UCI Machine Learning Repository; diakses melalui Kaggle. Data tidak diperoleh melalui web scraping.")

    filtered = data[data["G3"].between(*grade_range)].copy()
    if subject != "Semua":
        filtered = filtered[filtered[SUBJECT_COL] == subject]
    if filtered.empty:
        st.warning("Tidak ada data pada filter tersebut. Ubah rentang nilai atau mata pelajaran.")
        return

    tab_overview, tab_eda, tab_model, tab_data = st.tabs(["Ringkasan", "EDA", "Model & Prediksi", "Data"])
    with tab_overview:
        render_overview(filtered, source)
    with tab_eda:
        render_eda(filtered)
    with tab_model:
        render_model(filtered)
    with tab_data:
        render_data(filtered)


if __name__ == "__main__":
    main()
