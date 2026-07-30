"""Dashboard Streamlit – Analisis Performa Akademik Siswa.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

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
      .small-note {color:#64748b; font-size:.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
SUBJECT_COL = "Mata Pelajaran"
SUBJECTS = {"Matematika": "student-mat.csv", "Bahasa Portugis": "student-por.csv"}


def make_demo_data(subject: str, n: int, seed: int) -> pd.DataFrame:
    """Data demonstrasi agar dashboard tetap tampil sebelum CSV asli diunggah."""
    rng = np.random.default_rng(seed)
    age = rng.integers(15, 22, n)
    g1 = np.clip(np.rint(rng.normal(11.2 if subject == "Matematika" else 12.0, 3.0, n)), 0, 20).astype(int)
    failures = rng.choice([0, 1, 2, 3], size=n, p=[0.70, 0.18, 0.08, 0.04])
    study = rng.integers(1, 5, n)
    dalc = rng.integers(1, 6, n)
    walc = np.maximum(dalc, rng.integers(1, 6, n))
    noise = rng.normal(0, 2.4, n)
    g3 = np.clip(np.rint(2.1 + 0.95 * g1 - 1.15 * failures + 0.35 * study - 0.25 * dalc + noise), 0, 20).astype(int)
    g2 = np.clip(np.rint(g3 + rng.normal(0, 1.6, n)), 0, 20).astype(int)
    return pd.DataFrame(
        {
            "school": rng.choice(["GP", "MS"], n, p=[0.87, 0.13]),
            "sex": rng.choice(["F", "M"], n),
            "age": age,
            "address": rng.choice(["U", "R"], n, p=[0.78, 0.22]),
            "famsize": rng.choice(["LE3", "GT3"], n, p=[0.28, 0.72]),
            "Pstatus": rng.choice(["T", "A"], n, p=[0.90, 0.10]),
            "Medu": rng.integers(0, 5, n),
            "Fedu": rng.integers(0, 5, n),
            "traveltime": rng.integers(1, 5, n),
            "studytime": study,
            "failures": failures,
            "schoolsup": rng.choice(["yes", "no"], n, p=[0.13, 0.87]),
            "famsup": rng.choice(["yes", "no"], n, p=[0.61, 0.39]),
            "paid": rng.choice(["yes", "no"], n, p=[0.40, 0.60]),
            "activities": rng.choice(["yes", "no"], n),
            "higher": rng.choice(["yes", "no"], n, p=[0.88, 0.12]),
            "internet": rng.choice(["yes", "no"], n, p=[0.82, 0.18]),
            "romantic": rng.choice(["yes", "no"], n, p=[0.34, 0.66]),
            "famrel": rng.integers(1, 6, n),
            "freetime": rng.integers(1, 6, n),
            "goout": rng.integers(1, 6, n),
            "Dalc": dalc,
            "Walc": walc,
            "health": rng.integers(1, 6, n),
            "absences": rng.poisson(5, n),
            "G1": g1,
            "G2": g2,
            "G3": g3,
        }
    )


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    frames = []
    for subject, n, seed in [("Matematika", 395, 42), ("Bahasa Portugis", 649, 99)]:
        frame = make_demo_data(subject, n, seed)
        frame[SUBJECT_COL] = subject
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_dataset(uploaded_file=None, local_path: Path | None = None) -> pd.DataFrame:
    """Membaca CSV UCI/Kaggle dengan pemisah titik koma atau koma secara otomatis."""
    source = BytesIO(uploaded_file.getvalue()) if uploaded_file is not None else local_path
    return pd.read_csv(source, sep=None, engine="python")


def load_data(mat_upload, por_upload) -> tuple[pd.DataFrame, str]:
    """Prioritas: unggahan dashboard -> folder data -> mode demonstrasi."""
    try:
        if mat_upload is not None and por_upload is not None:
            mat = read_dataset(uploaded_file=mat_upload)
            por = read_dataset(uploaded_file=por_upload)
            source = "Dataset asli dari unggahan"
        elif (DATA_DIR / SUBJECTS["Matematika"]).exists() and (DATA_DIR / SUBJECTS["Bahasa Portugis"]).exists():
            mat = read_dataset(local_path=DATA_DIR / SUBJECTS["Matematika"])
            por = read_dataset(local_path=DATA_DIR / SUBJECTS["Bahasa Portugis"])
            source = "Dataset asli dari folder data"
        else:
            return demo_data().copy(), "Mode demonstrasi (data sintetis)"

        mat[SUBJECT_COL] = "Matematika"
        por[SUBJECT_COL] = "Bahasa Portugis"
        data = pd.concat([mat, por], ignore_index=True, sort=False)
        if "G3" not in data.columns:
            raise ValueError("Kolom G3 tidak ditemukan pada file CSV.")
        return data, source
    except Exception as exc:
        st.sidebar.error(f"CSV tidak dapat dibaca: {exc}")
        return demo_data().copy(), "Mode demonstrasi karena pembacaan CSV gagal"


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
    if source.startswith("Mode demonstrasi"):
        st.warning("Dashboard tampil dengan data sintetis. Unggah dua CSV asli pada sidebar untuk hasil analisis yang valid.")
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


def render_model(data: pd.DataFrame, source: str):
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
        st.header("Sumber data")
        st.caption("Unggah kedua file agar dashboard menggunakan data asli.")
        mat_upload = st.file_uploader("student-mat.csv", type=["csv"], key="mat")
        por_upload = st.file_uploader("student-por.csv", type=["csv"], key="por")
        st.divider()
        st.header("Filter")
    data, source = load_data(mat_upload, por_upload)
    with st.sidebar:
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
        render_model(filtered, source)
    with tab_data:
        render_data(filtered)


if __name__ == "__main__":
    main()
