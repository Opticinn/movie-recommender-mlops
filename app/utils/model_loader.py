import os
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from surprise import SVD, Dataset, Reader

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SVD_MODEL_PATH = MODEL_DIR / "svd_model_light.pkl"  # ✅ Format .pkl
EMBEDDINGS_PATH = MODEL_DIR / "sbert_embeddings.npy"  # ✅ Format .npy
HF_REPO_ID = "Hleanz/movie-recommender-models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Functions (TERPISAH JELAS) ─────────────────────────────────────────
def load_model(path: Path):
    """Load model .pkl menggunakan pickle"""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings .npy menggunakan numpy (AMAN, tanpa dependency)"""
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

# ── SVD ──────────────────────────────────────────────────────────────────────
def get_svd_model(force_retrain: bool = False) -> dict:
    if SVD_MODEL_PATH.exists() and not force_retrain:
        print("⚡ Loading SVD model from cache...")
        return load_model(SVD_MODEL_PATH)  # ✅ Pakai pickle.load()

    try:
        print("📥 Downloading SVD model from HuggingFace Hub...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename="svd_model_light.pkl",
            local_dir=MODEL_DIR, token=os.getenv("HF_TOKEN"),
        )
        return load_model(Path(downloaded_path))
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download SVD: {e}")

# ── SBERT Embeddings ─────────────────────────────────────────────────────────
def get_sbert_embeddings(force_recompute: bool = False) -> np.ndarray:
    if EMBEDDINGS_PATH.exists() and not force_recompute:
        print("⚡ Loading embeddings from cache...")
        return load_embeddings(EMBEDDINGS_PATH)  # ✅ Pakai np.load()

    try:
        print("📥 Downloading SBERT embeddings from HuggingFace Hub...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename="sbert_embeddings.npy",
            local_dir=MODEL_DIR, token=os.getenv("HF_TOKEN"),
        )
        return load_embeddings(Path(downloaded_path))
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download SBERT embeddings: {e}")

# ── Hybrid Recommendations ───────────────────────────────────────────────────
def get_hybrid_recommendations(
    user_id: int, movies_df: pd.DataFrame, svd_model: dict, 
    embeddings: np.ndarray, top_n: int = 10, alpha: float = 0.6
) -> pd.DataFrame:
    movie_ids = movies_df["movieId"].tolist()
    results = []

    for idx, movie_id in enumerate(movie_ids):
        svd_score = predict_svd_light(svd_model, user_id, movie_id)
        emb = embeddings[idx]
        similarities = [
            float(np.dot(emb, embeddings[j]) / (np.linalg.norm(emb) * np.linalg.norm(embeddings[j]) or 1e-9))
            for j in range(len(embeddings)) if j != idx
        ]
        content_score = float(np.mean(similarities)) if similarities else 0.0
        hybrid_score = alpha * svd_score + (1 - alpha) * content_score
        results.append({
            "movieId": movie_id, "title": movies_df.iloc[idx]["title"],
            "svd_score": round(svd_score, 4), "content_score": round(content_score, 4),
            "hybrid_score": round(hybrid_score, 4),
        })

    return pd.DataFrame(results).sort_values("hybrid_score", ascending=False).head(top_n)