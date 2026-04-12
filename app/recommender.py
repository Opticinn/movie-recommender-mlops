# app/recommender.py
# Streamlit app for movie recommendation system

import sys
import os
import time
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_client import get_client, insert_prediction_log

# ── 1. Page configuration ──────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── 2. Load movie data from Supabase ──────────────────────
@st.cache_data(ttl=3600)
def load_movies() -> pd.DataFrame:
    try:
        client = get_client()
        response = client.table("cache_ratings").select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"❌ Failed to load movie data: {e}")
        return pd.DataFrame()


# ── 3. Genre similarity function ──────────────────────────
def genre_similarity(genres_a: list, genres_b: list) -> float:
    """
    Measure similarity between two movies based on genre.
    Uses Jaccard Similarity: intersection / union
    Returns 0.0 (no match) to 1.0 (identical)
    """
    if not genres_a or not genres_b:
        return 0.0

    set_a = set(genres_a)
    set_b = set(genres_b)

    intersection = set_a & set_b
    union = set_a | set_b

    return len(intersection) / len(union)


def get_recommendations(
    input_movie: pd.Series,
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Find the most similar movies to the input movie.
    Returns a DataFrame of top_n recommendations.
    """
    input_genres = input_movie["genre_names"]

    df = df.copy()
    df["similarity"] = df["genre_names"].apply(
        lambda genres: genre_similarity(input_genres, genres)
    )

    df = df[df["movie_id"] != input_movie["movie_id"]]
    df = df[df["similarity"] > 0]
    df = df.sort_values(
        by=["similarity", "rating"],
        ascending=[False, False]
    )

    return df.head(top_n)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

def get_poster_url(poster_path: str) -> str | None:
    """
    Konversi poster_path → URL lengkap.
    Return None kalau poster tidak tersedia.
    """
    if poster_path and str(poster_path) != "None":
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    return None

# ── 4. Main UI ─────────────────────────────────────────────
def main():
    st.title("🎬 Movie Recommender")
    st.caption("Discover movies similar to your favorites!")

    df = load_movies()
    if df.empty:
        st.error("No movie data found. Please run fetch_ratings.py first.")
        return

    st.sidebar.header("🔍 Search Movie")

    # ── Search box ─────────────────────────────────────────
    search_query = st.sidebar.text_input(
        "Type a movie title:",
        placeholder="e.g. Inception"
    )

    if search_query:
        filtered_df = df[
            df["title"].str.contains(search_query, case=False, na=False)
        ]
    else:
        filtered_df = df

    if filtered_df.empty:
        st.sidebar.warning("No movies found. Try a different keyword.")
        return

    # ── Selectbox ──────────────────────────────────────────
    movie_titles = filtered_df["title"].tolist()
    selected_title = st.sidebar.selectbox("Select a movie:", movie_titles)

    input_movie = filtered_df[filtered_df["title"] == selected_title].iloc[0]

    # ── Selected movie info ────────────────────────────────
    st.subheader("🎯 Your selected movie:")

    poster_url = get_poster_url(input_movie.get("poster_path"))

    col_poster, col_info = st.columns([1, 3])

    with col_poster:
        if poster_url:
            st.image(poster_url, width=150)
        else:
            st.caption("No poster available")

    with col_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Title", input_movie["title"])
        col2.metric("Rating", f"{input_movie['rating']}/10")
        col3.metric("Year", input_movie["release_year"] or "N/A")
        st.caption(f"Genre: {', '.join(input_movie['genre_names'] or [])}")

    st.divider()

    # ── Recommend button ───────────────────────────────────
    if st.button("🚀 Find Recommendations!", type="primary"):

        with st.spinner("Finding similar movies..."):
            start = time.time()
            recommendations = get_recommendations(input_movie, df)
            latency_ms = int((time.time() - start) * 1000)

        if recommendations.empty:
            st.warning("No recommendations found.")
            return

        # ── Display recommendations ────────────────────────
        st.subheader(f"🎬 Top 10 Recommendations for '{selected_title}':")

        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            with st.expander(f"{i}. {movie['title']} ⭐ {movie['rating']}"):
                col_poster, col_detail = st.columns([1, 3])

                with col_poster:
                    poster_url = get_poster_url(movie.get("poster_path"))
                    if poster_url:
                        st.image(poster_url, width=120)
                    else:
                        st.caption("No poster")

                with col_detail:
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Year:** {movie['release_year'] or 'N/A'}")
                    c2.write(f"**Rating:** {movie['rating']}/10")
                    c3.write(f"**Similarity:** {movie['similarity']:.0%}")
                    st.write(f"**Genre:** {', '.join(movie['genre_names'] or [])}")

        # ── Log to database ────────────────────────────────
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