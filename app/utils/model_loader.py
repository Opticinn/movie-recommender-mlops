import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SVD_MODEL_PATH = MODEL_DIR / "svd_model_light.pkl" 
EMBEDDINGS_PATH = MODEL_DIR / "sbert_embeddings.npy"
DATA_PATH = BASE_DIR / "data" / "ratings.csv"

# HuggingFace Hub config
HF_REPO_ID = "Hleanz/movie-recommender-models"  

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def save_model(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"💾 Saved: {path.name}")


def load_model(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)
    
def predict_svd_light(model_data: dict, uid: int, iid: int) -> float:
    """Prediksi rating menggunakan model ringan (tanpa trainset)"""
    inner_u = model_data['raw2inner_uid'].get(uid)
    inner_i = model_data['raw2inner_iid'].get(iid)
    est = model_data['global_mean']
    if inner_u is not None: est += model_data['bu'][inner_u]
    if inner_i is not None: est += model_data['bi'][inner_i]
    if inner_u is not None and inner_i is not None:
        est += float(np.dot(model_data['pu'][inner_u], model_data['qi'][inner_i]))
    return est


# ── Data ─────────────────────────────────────────────────────────────────────
def load_movielens():
    df = pd.read_csv(DATA_PATH)
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)


# ── SVD ──────────────────────────────────────────────────────────────────────
def train_svd(ratings):
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    trainset = ratings.build_full_trainset()
    svd.fit(trainset)
    return svd


def get_svd_model(force_retrain: bool = False) -> dict:
    """Load SVD model (lightweight dict): cache → HF Hub → train (fallback)."""
    if SVD_MODEL_PATH.exists() and not force_retrain:
        print("⚡ Loading SVD model from cache...")
        return load_model(SVD_MODEL_PATH)

    # Download dari HF Hub
    try:
        print("📥 Downloading SVD model from HuggingFace Hub...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="svd_model_light.pkl",
            local_dir=MODEL_DIR,
            token=os.getenv("HF_TOKEN"),
        )
        print("✅ SVD model downloaded!")
        return load_model(Path(downloaded_path))

    except Exception as e:
        raise RuntimeError(
            f"❌ Gagal download SVD model dari HF Hub: {e}\n"
            "Pastikan HF_TOKEN sudah di-set di Streamlit Secrets."
    )


# ── SBERT Embeddings ─────────────────────────────────────────────────────────
def compute_embeddings(movies_df: pd.DataFrame):
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    titles = movies_df["title"].tolist()
    embeddings = sbert.encode(titles, show_progress_bar=True)
    return embeddings

# ── SBERT Embeddings ─────────────────────────────────────────────────────────
def get_sbert_embeddings(force_recompute: bool = False) -> np.ndarray:
    if EMBEDDINGS_PATH.exists() and not force_recompute:
        print("⚡ Loading embeddings from cache...")
        return load_model(EMBEDDINGS_PATH)  # ✅ PAKAI np.load

    try:
        print("📥 Downloading SBERT embeddings from HuggingFace Hub...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="sbert_embeddings.npy",  # ✅ Pastikan nama file .npy
            local_dir=MODEL_DIR,
            token=os.getenv("HF_TOKEN"),
        )
        print("✅ SBERT embeddings downloaded!")
        return load_model(Path(downloaded_path))  # ✅ PAKAI np.load
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download SBERT embeddings: {e}")


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm) if norm != 0 else 0.0


# ── Hybrid Recommendations ───────────────────────────────────────────────────
def get_hybrid_recommendations(
    user_id: int,
    movies_df: pd.DataFrame,
    svd_model: dict,
    embeddings: np.ndarray,
    top_n: int = 10,
    alpha: float = 0.6,
) -> pd.DataFrame:
    """
    Combine SVD collaborative filtering + SBERT content similarity.
    alpha controls SVD weight (1-alpha = SBERT weight).
    """
    movie_ids = movies_df["movieId"].tolist()
    results = []

    for idx, movie_id in enumerate(movie_ids):
        # SVD score
        svd_score = predict_svd_light(svd_model, user_id, movie_id)

        # SBERT similarity (average similarity to all other movies)
        emb = embeddings[idx]
        similarities = [
            cosine_similarity(emb, embeddings[j])
            for j in range(len(embeddings))
            if j != idx
        ]
        content_score = float(np.mean(similarities)) if similarities else 0.0

        # Hybrid score
        hybrid_score = alpha * svd_score + (1 - alpha) * content_score
        results.append(
            {
                "movieId": movie_id,
                "title": movies_df.iloc[idx]["title"],
                "svd_score": round(svd_score, 4),
                "content_score": round(content_score, 4),
                "hybrid_score": round(hybrid_score, 4),
            }
        )

    result_df = pd.DataFrame(results)
    return result_df.sort_values("hybrid_score", ascending=False).head(top_n)