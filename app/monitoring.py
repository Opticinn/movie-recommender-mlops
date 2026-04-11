# app/monitoring.py
# Dashboard monitoring data drift menggunakan Streamlit
# Membandingkan data harian vs baseline metrics

import sys
import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_client import get_client, get_baseline_metrics, insert_prediction_log

# ── 1. Konfigurasi halaman ─────────────────────────────────
st.set_page_config(
    page_title="MLOps Monitor",
    page_icon="📊",
    layout="wide"
)

# ── 2. Load data ───────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache 5 menit — lebih sering refresh vs recommender
def load_prediction_logs(days: int = 7) -> pd.DataFrame:
    """
    Ambil prediction logs dari N hari terakhir.
    """
    try:
        client = get_client()
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        response = (
            client.table("prediction_logs")
            .select("*")
            .gte("timestamp", since)  # gte = greater than or equal
            .execute()
        )

        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"❌ Gagal load prediction logs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_cache_ratings() -> pd.DataFrame:
    """
    Ambil semua data film dari cache_ratings.
    """
    try:
        client = get_client()
        response = client.table("cache_ratings").select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"❌ Gagal load cache ratings: {e}")
        return pd.DataFrame()
    
    
# ── 3. Fungsi deteksi drift ────────────────────────────────

def detect_rating_drift(
    df_ratings: pd.DataFrame,
    baseline: dict
) -> dict:
    """
    Bandingkan mean rating harian vs baseline.
    Return dict berisi status & detail drift.
    """
    baseline_mean = float(baseline.get("mean_rating", 3.5))
    threshold = float(baseline.get("drift_threshold_rating", 0.3))

    current_mean = df_ratings["rating"].mean()
    drift = abs(current_mean - baseline_mean)

    if drift >= threshold:
        severity = "red"
        status = f"🔴 DRIFT TERDETEKSI"
    elif drift >= threshold * 0.7:  # 70% dari threshold = warning
        severity = "yellow"
        status = f"🟡 MENDEKATI THRESHOLD"
    else:
        severity = "green"
        status = f"🟢 NORMAL"

    return {
        "status": status,
        "severity": severity,
        "baseline_mean": baseline_mean,
        "current_mean": round(current_mean, 3),
        "drift": round(drift, 3),
        "threshold": threshold
    }


def detect_latency_drift(
    df_logs: pd.DataFrame,
    baseline: dict
) -> dict:
    """
    Cek apakah rata-rata latency melebihi threshold.
    """
    threshold = float(baseline.get("latency_threshold_ms", 2000))

    if df_logs.empty or "latency_ms" not in df_logs.columns:
        return {"status": "⚪ TIDAK ADA DATA", "severity": "grey"}

    current_mean = df_logs["latency_ms"].dropna().mean()
    drift = current_mean - threshold

    if current_mean >= threshold:
        severity = "red"
        status = "🔴 LATENCY TINGGI"
    elif current_mean >= threshold * 0.7:
        severity = "yellow"
        status = "🟡 LATENCY MENDEKATI BATAS"
    else:
        severity = "green"
        status = "🟢 NORMAL"

    return {
        "status": status,
        "severity": severity,
        "current_mean_ms": round(current_mean, 1),
        "threshold_ms": threshold,
        "drift_ms": round(drift, 1)
    }


def detect_genre_drift(
    df_ratings: pd.DataFrame,
    baseline_top_genres: list = None
) -> dict:
    """
    Cek distribusi genre harian vs baseline.
    """
    # Explode genre_names (dari list per baris → satu genre per baris)
    genres_exploded = df_ratings["genre_names"].explode()
    current_top = genres_exploded.value_counts().head(5).index.tolist()

    if baseline_top_genres is None:
        baseline_top_genres = ["Drama", "Comedy", "Action", "Thriller", "Romance"]

    # Hitung overlap top 5 genre
    overlap = len(set(current_top) & set(baseline_top_genres))
    overlap_pct = overlap / 5

    if overlap_pct <= 0.4:
        severity = "red"
        status = "🔴 GENRE SHIFT SIGNIFIKAN"
    elif overlap_pct <= 0.6:
        severity = "yellow"
        status = "🟡 GENRE BERGESER"
    else:
        severity = "green"
        status = "🟢 NORMAL"

    return {
        "status": status,
        "severity": severity,
        "baseline_top_genres": baseline_top_genres,
        "current_top_genres": current_top,
        "overlap_pct": f"{overlap_pct:.0%}"
    }
    
# ── 4. Main Dashboard ──────────────────────────────────────
def main():
    st.title("📊 MLOps Monitoring Dashboard")
    st.caption("Real-time monitoring untuk Movie Recommender System")

    # ── Sidebar: filter waktu ──────────────────────────────
    st.sidebar.header("⚙️ Settings")
    days = st.sidebar.slider("Rentang waktu (hari):", 1, 30, 7)
    st.sidebar.divider()

    # ── Load data ──────────────────────────────────────────
    baseline = get_baseline_metrics()
    df_logs = load_prediction_logs(days=days)
    df_ratings = load_cache_ratings()

    if baseline:
        st.sidebar.success("✅ Baseline loaded")
    else:
        st.sidebar.error("❌ Baseline tidak ditemukan")

    # ── Section 1: Overview metrics ────────────────────────
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Film di Cache",
        len(df_ratings),
        delta=f"{len(df_ratings) - int(float(baseline.get('total_films', 500)))} dari target"
    )
    col2.metric(
        "Total Prediksi",
        len(df_logs) if not df_logs.empty else 0
    )
    col3.metric(
        "Mean Rating Cache",
        f"{df_ratings['rating'].mean():.2f}" if not df_ratings.empty else "N/A"
    )
    col4.metric(
        "Baseline Mean Rating",
        baseline.get("mean_rating", "N/A")
    )

    st.divider()

    # ── Section 2: Drift Detection ─────────────────────────
    st.subheader("🚨 Drift Detection")

    col1, col2, col3 = st.columns(3)

    # Rating drift
    with col1:
        st.markdown("**Rating Drift**")
        if not df_ratings.empty:
            result = detect_rating_drift(df_ratings, baseline)
            st.markdown(f"### {result['status']}")
            st.write(f"Baseline: `{result['baseline_mean']}`")
            st.write(f"Current:  `{result['current_mean']}`")
            st.write(f"Drift:    `{result['drift']}` (threshold: {result['threshold']})")
        else:
            st.warning("Tidak ada data rating")

    # Latency drift
    with col2:
        st.markdown("**Latency Drift**")
        if not df_logs.empty:
            result = detect_latency_drift(df_logs, baseline)
            st.markdown(f"### {result['status']}")
            if result["severity"] != "grey":
                st.write(f"Current:   `{result['current_mean_ms']} ms`")
                st.write(f"Threshold: `{result['threshold_ms']} ms`")
        else:
            st.info("⚪ Belum ada prediction logs")

    # Genre drift
    with col3:
        st.markdown("**Genre Drift**")
        if not df_ratings.empty:
            result = detect_genre_drift(df_ratings)
            st.markdown(f"### {result['status']}")
            st.write(f"Baseline: `{', '.join(result['baseline_top_genres'])}`")
            st.write(f"Current:  `{', '.join(result['current_top_genres'])}`")
            st.write(f"Overlap:  `{result['overlap_pct']}`")

    st.divider()

    # ── Section 3: Charts ──────────────────────────────────
    st.subheader("📊 Distribusi Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribusi Rating Film**")
        if not df_ratings.empty:
            hist_data = df_ratings["rating"].value_counts().sort_index()
            st.bar_chart(hist_data)

    with col2:
        st.markdown("**Top 10 Genre**")
        if not df_ratings.empty:
            genre_counts = (
                df_ratings["genre_names"]
                .explode()
                .value_counts()
                .head(10)
            )
            st.bar_chart(genre_counts)

    # ── Section 4: Raw logs ────────────────────────────────
    st.divider()
    st.subheader("📋 Recent Prediction Logs")

    if df_logs.empty:
        st.info("Belum ada prediction logs dalam rentang waktu ini.")
        st.caption("💡 Coba buat beberapa rekomendasi di app utama dulu!")
    else:
        # Tampilkan kolom yang relevan saja
        cols = ["timestamp", "input_movie_title", "recommended_movies",
                "latency_ms", "api_source", "model_version"]
        cols = [c for c in cols if c in df_logs.columns]
        st.dataframe(df_logs[cols], use_container_width=True)


if __name__ == "__main__":
    main()