# app/recommender.py
# Streamlit app untuk sistem rekomendasi film

import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.db_client import get_client, insert_prediction_log

# ── 1. Konfigurasi halaman ─────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── 2. Load data film dari Supabase ───────────────────────
@st.cache_data(ttl=3600)  # Cache 1 jam — tidak perlu fetch ulang tiap interaksi
def load_movies() -> pd.DataFrame:
    """
    Ambil semua film dari cache_ratings dan return sebagai DataFrame.
    """
    try:
        client = get_client()
        response = client.table("cache_ratings").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data film: {e}")
        return pd.DataFrame()
    
    
# ── 3. Fungsi hitung similarity berbasis genre ─────────────
def genre_similarity(genres_a: list, genres_b: list) -> float:
    """
    Hitung seberapa mirip dua film berdasarkan genre.
    Menggunakan Jaccard Similarity: irisan / gabungan
    Return nilai 0.0 (tidak mirip) sampai 1.0 (identik)
    """
    if not genres_a or not genres_b:
        return 0.0

    set_a = set(genres_a)
    set_b = set(genres_b)

    intersection = set_a & set_b  # genre yang sama
    union = set_a | set_b         # semua genre gabungan

    return len(intersection) / len(union)


def get_recommendations(
    input_movie: pd.Series,
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Cari film paling mirip dengan input_movie.
    Return DataFrame berisi top_n rekomendasi.
    """
    input_genres = input_movie["genre_names"]

    # Hitung similarity semua film terhadap input
    df = df.copy()
    df["similarity"] = df["genre_names"].apply(
        lambda genres: genre_similarity(input_genres, genres)
    )

    # Exclude film input sendiri
    df = df[df["movie_id"] != input_movie["movie_id"]]

    # Filter hanya yang ada kemiripan genre
    df = df[df["similarity"] > 0]

    # Sort: similarity dulu, lalu rating sebagai tiebreaker
    df = df.sort_values(
        by=["similarity", "rating"],
        ascending=[False, False]
    )

    return df.head(top_n)


# ── 3. Fungsi hitung similarity berbasis genre ─────────────
def genre_similarity(genres_a: list, genres_b: list) -> float:
    """
    Hitung seberapa mirip dua film berdasarkan genre.
    Menggunakan Jaccard Similarity: irisan / gabungan
    Return nilai 0.0 (tidak mirip) sampai 1.0 (identik)
    """
    if not genres_a or not genres_b:
        return 0.0

    set_a = set(genres_a)
    set_b = set(genres_b)

    intersection = set_a & set_b  # genre yang sama
    union = set_a | set_b         # semua genre gabungan

    return len(intersection) / len(union)


def get_recommendations(
    input_movie: pd.Series,
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Cari film paling mirip dengan input_movie.
    Return DataFrame berisi top_n rekomendasi.
    """
    input_genres = input_movie["genre_names"]

    # Hitung similarity semua film terhadap input
    df = df.copy()
    df["similarity"] = df["genre_names"].apply(
        lambda genres: genre_similarity(input_genres, genres)
    )

    # Exclude film input sendiri
    df = df[df["movie_id"] != input_movie["movie_id"]]

    # Filter hanya yang ada kemiripan genre
    df = df[df["similarity"] > 0]

    # Sort: similarity dulu, lalu rating sebagai tiebreaker
    df = df.sort_values(
        by=["similarity", "rating"],
        ascending=[False, False]
    )

    return df.head(top_n)

# ── 4. Main UI ─────────────────────────────────────────────
def main():
    st.title("🎬 Movie Recommender")
    st.caption("Temukan film yang mirip dengan favoritmu!")

    # Load data
    df = load_movies()
    if df.empty:
        st.error("Tidak ada data film. Jalankan fetch_ratings.py terlebih dahulu.")
        return

    st.sidebar.header("🔍 Cari Film")

    # ── Search box ─────────────────────────────────────────
    search_query = st.sidebar.text_input(
        "Ketik judul film:",
        placeholder="contoh: Inception"
    )

    # Filter film berdasarkan search query
    if search_query:
        filtered_df = df[
            df["title"].str.contains(search_query, case=False, na=False)
        ]
    else:
        filtered_df = df

    if filtered_df.empty:
        st.sidebar.warning("Film tidak ditemukan. Coba kata lain.")
        return

    # ── Selectbox dari hasil filter ────────────────────────
    movie_titles = filtered_df["title"].tolist()
    selected_title = st.sidebar.selectbox("Pilih film:", movie_titles)

    # Ambil data film yang dipilih
    input_movie = filtered_df[filtered_df["title"] == selected_title].iloc[0]

    # ── Info film yang dipilih ─────────────────────────────
    st.subheader("🎯 Film yang kamu pilih:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Judul", input_movie["title"])
    col2.metric("Rating", f"{input_movie['rating']}/10")
    col3.metric("Tahun", input_movie["release_year"] or "N/A")
    st.caption(f"Genre: {', '.join(input_movie['genre_names'] or [])}")

    st.divider()

    # ── Tombol rekomendasi ─────────────────────────────────
    if st.button("🚀 Cari Rekomendasi!", type="primary"):

        with st.spinner("Mencari film yang mirip..."):
            import time
            start = time.time()

            recommendations = get_recommendations(input_movie, df)

            latency_ms = int((time.time() - start) * 1000)

        if recommendations.empty:
            st.warning("Tidak ada rekomendasi ditemukan.")
            return

        # ── Tampilkan rekomendasi ──────────────────────────
        st.subheader(f"🎬 10 Film Rekomendasi untuk '{selected_title}':")

        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            with st.expander(f"{i}. {movie['title']} ⭐ {movie['rating']}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Tahun:** {movie['release_year'] or 'N/A'}")
                c2.write(f"**Rating:** {movie['rating']}/10")
                c3.write(f"**Similarity:** {movie['similarity']:.0%}")
                st.write(f"**Genre:** {', '.join(movie['genre_names'] or [])}")

        # ── Log ke database ────────────────────────────────
        insert_prediction_log(
            session_id=st.session_state.get("session_id", "streamlit-session"),
            input_movie_id=int(input_movie["movie_id"]),
            input_movie_title=input_movie["title"],
            input_movie_genres=input_movie["genre_names"],
            recommended_movies=recommendations["title"].tolist(),
            recommended_genres=recommendations["genre_names"].tolist(),
            latency_ms=latency_ms,
            api_source="tmdb"
        )

        st.caption(f"⚡ Latency: {latency_ms}ms")


if __name__ == "__main__":
    main()