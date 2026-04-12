# app/utils/model_loader.py
# Load, train, dan cache hybrid recommendation model
# SVD (Collaborative Filtering) + SBERT (Content-Based)

import sys
import os
import pickle
import pandas as pd
from pathlib import Path

# Path ke dataset dan model
DATA_DIR  = Path("data")
MODEL_DIR = Path("model")

RATINGS_PATH = DATA_DIR / "ratings.csv"
MOVIES_PATH  = DATA_DIR / "movies.csv"
SVD_MODEL_PATH = MODEL_DIR / "svd_model.pkl"

# ── 1. Load MovieLens dataset ──────────────────────────────
def load_movielens(min_ratings: int = 50) -> pd.DataFrame:
    """
    Load MovieLens ratings.csv dan filter film
    yang punya minimal min_ratings rating.
    Return DataFrame siap pakai untuk training SVD.
    """
    print("📂 Loading MovieLens dataset...")

    ratings = pd.read_csv(RATINGS_PATH)

    print(f"   Raw ratings: {len(ratings):,} rows")

    # Filter film dengan minimal 50 rating — buang film yang terlalu obscure
    movie_counts = ratings["movieId"].value_counts()
    popular_movies = movie_counts[movie_counts >= min_ratings].index
    ratings = ratings[ratings["movieId"].isin(popular_movies)]

    print(f"   Filtered ratings: {len(ratings):,} rows")
    print(f"   Unique movies: {ratings['movieId'].nunique():,}")
    print(f"   Unique users: {ratings['userId'].nunique():,}")

    return ratings


from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# ── 2. Train SVD model ─────────────────────────────────────
def train_svd(ratings: pd.DataFrame) -> SVD:
    """
    Train SVD model dari MovieLens ratings.
    Return trained SVD model.
    """
    print("\n🧠 Training SVD model...")

    # Konversi DataFrame → Surprise Dataset
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    # Split train/test 80/20
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train SVD
    svd = SVD(
        n_factors=100,    # jumlah latent factors
        n_epochs=20,      # iterasi training
        lr_all=0.005,     # learning rate
        reg_all=0.02,     # regularisasi — cegah overfitting
        random_state=42
    )
    svd.fit(trainset)

    # Evaluasi model
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"   ✅ SVD trained! RMSE: {rmse:.4f}")

    return svd


# ── 3. Simpan & Load model dari disk ──────────────────────
def save_model(model, path: Path) -> None:
    """Simpan model ke disk pakai pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"   💾 Model saved to {path}")


def load_model(path: Path):
    """Load model dari disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_svd_model(force_retrain: bool = False) -> SVD:
    """
    Load SVD model dari disk kalau sudah ada.
    Train ulang kalau belum ada atau force_retrain=True.
    """
    if SVD_MODEL_PATH.exists() and not force_retrain:
        print("⚡ Loading SVD model from cache...")
        return load_model(SVD_MODEL_PATH)

    # Train dari awal
    ratings = load_movielens()
    svd = train_svd(ratings)
    save_model(svd, SVD_MODEL_PATH)
    return svd


from sentence_transformers import SentenceTransformer
import numpy as np

SBERT_MODEL_PATH = MODEL_DIR / "sbert_embeddings.pkl"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Model kecil tapi akurat, ~80MB

# ── 4. Load sinopsis dari TMDB ─────────────────────────────
def fetch_movie_overviews(movie_ids: list) -> dict:
    """
    Fetch sinopsis film dari TMDB API.
    Return dict {movie_id: overview}
    """
    import requests
    import time
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("TMDB_API_KEY")
    overviews = {}

    print(f"\n📝 Fetching overviews for {len(movie_ids)} movies...")

    for i, movie_id in enumerate(movie_ids):
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            response = requests.get(
                url,
                params={"api_key": api_key},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            overview = data.get("overview", "")

            # Skip film tanpa sinopsis
            if overview:
                overviews[movie_id] = overview

            # Progress setiap 50 film
            if (i + 1) % 50 == 0:
                print(f"   {i + 1}/{len(movie_ids)} fetched...")

            time.sleep(0.1)  # Hormati rate limit TMDB

        except Exception as e:
            print(f"   ⚠️ Failed for movie_id {movie_id}: {e}")
            continue

    print(f"   ✅ Got overviews for {len(overviews)} movies")
    return overviews


# ── 5. Generate SBERT embeddings ──────────────────────────
def get_sbert_embeddings(
    movie_ids: list,
    force_regenerate: bool = False
) -> tuple[dict, SentenceTransformer]:
    """
    Generate SBERT embeddings untuk setiap film.
    Return (embeddings_dict, sbert_model)
    embeddings_dict = {movie_id: embedding_vector}
    """
    if SBERT_MODEL_PATH.exists() and not force_regenerate:
        print("⚡ Loading SBERT embeddings from cache...")
        return load_model(SBERT_MODEL_PATH)

    print(f"\n🤖 Loading SBERT model: {SBERT_MODEL_NAME}")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    # Fetch sinopsis dari TMDB
    overviews = fetch_movie_overviews(movie_ids)

    if not overviews:
        print("❌ No overviews found!")
        return {}, sbert

    # Generate embeddings
    print("\n🔢 Generating embeddings...")
    ids = list(overviews.keys())
    texts = list(overviews.values())

    embeddings = sbert.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Simpan sebagai dict {movie_id: embedding}
    embeddings_dict = {
        movie_id: embeddings[i]
        for i, movie_id in enumerate(ids)
    }

    # Cache ke disk
    save_model((embeddings_dict, sbert), SBERT_MODEL_PATH)
    print(f"   ✅ Generated {len(embeddings_dict)} embeddings")

    return embeddings_dict, sbert


# ── 6. Hitung cosine similarity ───────────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Hitung cosine similarity antara dua embedding vector.
    Return nilai -1.0 sampai 1.0 (semakin tinggi = semakin mirip)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ── 7. Hybrid recommendation ──────────────────────────────
def get_hybrid_recommendations(
    input_movie_id: int,
    candidate_movies: pd.DataFrame,
    svd_model: SVD,
    embeddings_dict: dict,
    user_id: int = 1,
    svd_weight: float = 0.4,
    content_weight: float = 0.6,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Hybrid recommendation: SVD + Content-Based (Genre + SBERT)
    
    Parameters:
    - input_movie_id  : TMDB movie ID film input
    - candidate_movies: DataFrame semua film dari cache_ratings
    - svd_model       : trained SVD model
    - embeddings_dict : SBERT embeddings {movie_id: vector}
    - user_id         : user ID untuk SVD (default 1 = dummy user)
    - svd_weight      : bobot SVD score (default 0.4)
    - content_weight  : bobot content score (default 0.6)
    - top_n           : jumlah rekomendasi
    """
    results = []

    # Ambil embedding film input
    input_embedding = embeddings_dict.get(input_movie_id)
    input_movie = candidate_movies[
        candidate_movies["movie_id"] == input_movie_id
    ].iloc[0]
    input_genres = set(input_movie["genre_names"] or [])

    for _, movie in candidate_movies.iterrows():
        movie_id = int(movie["movie_id"])

        # Skip film input sendiri
        if movie_id == input_movie_id:
            continue

        # ── SVD Score ─────────────────────────────────────
        try:
            svd_score = svd_model.predict(
                uid=user_id,
                iid=movie_id
            ).est  # est = estimated rating (0.5 - 5.0)

            # Normalisasi ke 0-1
            svd_score_normalized = (svd_score - 0.5) / 4.5

        except Exception:
            svd_score_normalized = 0.0

        # ── Content Score (Genre + SBERT) ─────────────────
        # Genre similarity (Jaccard)
        movie_genres = set(movie["genre_names"] or [])
        if input_genres and movie_genres:
            genre_sim = len(input_genres & movie_genres) / len(input_genres | movie_genres)
        else:
            genre_sim = 0.0

        # SBERT similarity
        movie_embedding = embeddings_dict.get(movie_id)
        if input_embedding is not None and movie_embedding is not None:
            sbert_sim = float(cosine_similarity(input_embedding, movie_embedding))
            sbert_sim = max(0.0, sbert_sim)  # clamp ke 0-1
        else:
            sbert_sim = 0.0

        # Gabungkan genre + SBERT (50/50)
        content_score = (genre_sim * 0.5) + (sbert_sim * 0.5)

        # ── Final Hybrid Score ────────────────────────────
        hybrid_score = (svd_score_normalized * svd_weight) + \
                       (content_score * content_weight)

        results.append({
            "movie_id": movie_id,
            "title": movie["title"],
            "rating": movie["rating"],
            "release_year": movie["release_year"],
            "genre_names": movie["genre_names"],
            "poster_path": movie.get("poster_path"),
            "svd_score": round(svd_score_normalized, 3),
            "content_score": round(content_score, 3),
            "hybrid_score": round(hybrid_score, 3)
        })

    # Sort by hybrid score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("hybrid_score", ascending=False)

    return results_df.head(top_n)
