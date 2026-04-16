import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

HF_REPO_ID = "Hleanz/movie-recommender-models"
SVD_MODEL_PATH = MODEL_DIR / "svd_model_light.pkl"
EMBEDDINGS_PATH = MODEL_DIR / "sbert_embeddings.npy"
MOVIE_IDS_PATH = MODEL_DIR / "sbert_movie_ids.npy"

def load_pickle(path: Path):
    """Fungsi aman untuk load file pickle (SVD Dict)"""
    with open(path, "rb") as f:
        return pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_svd_model(force_retrain: bool = False):
    if SVD_MODEL_PATH.exists() and not force_retrain:
        return load_pickle(SVD_MODEL_PATH)
    
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="svd_model_light.pkl",
            local_dir=MODEL_DIR,
            token=os.getenv("HF_TOKEN")
        )
        return load_pickle(SVD_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Gagal load SVD: {e}")

def get_sbert_embeddings():
    if not EMBEDDINGS_PATH.exists() or not MOVIE_IDS_PATH.exists():
        for filename in ["sbert_embeddings.npy", "sbert_movie_ids.npy"]:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=MODEL_DIR,
                token=os.getenv("HF_TOKEN")
            )
    
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    movie_ids = np.load(MOVIE_IDS_PATH, allow_pickle=False)
    return embeddings, movie_ids

def predict_svd_light(model_data: dict, uid: int, iid: int) -> float:
    """Prediksi rating menggunakan model ringan dengan pengecekan kunci yang aman"""
    
    # Coba cari kunci mean, mu, atau global_mean (tergantung hasil save pkl kamu)
    mean = model_data.get('mean', model_data.get('mu', model_data.get('global_mean', 0)))
    
    # Ambil bias user & item (default 0 jika tidak ada)
    bu = model_data.get('bu', {}).get(uid, 0)
    bi = model_data.get('bi', {}).get(iid, 0)
    
    # Ambil faktor latent (default array nol jika tidak ada)
    factor_size = model_data.get('factor_size', 100)
    pu = model_data.get('pu', {}).get(uid, np.zeros(factor_size))
    qi = model_data.get('qi', {}).get(iid, np.zeros(factor_size))
    
    # Hitung dot product (pu * qi) + mean + bias
    est = mean + bu + bi + np.dot(pu, qi)
    
    return float(np.clip(est, 0.5, 5.0))

def get_hybrid_recommendations(
    input_movie_id: int,
    candidate_movies: pd.DataFrame,
    svd_model: dict,
    embeddings_dict: dict,
    top_n: int = 10,
    alpha: float = 0.6,
) -> pd.DataFrame:
    # 1. Ambil embedding film input
    if input_movie_id not in embeddings_dict:
        return pd.DataFrame()
    
    input_vector = embeddings_dict[input_movie_id]
    user_id = 1 # Default user
    
    results = []
    for _, row in candidate_movies.iterrows():
        m_id = int(row['movie_id'])
        
        # Collaborative Score (SVD)
        svd_score = predict_svd_light(svd_model, user_id, m_id) / 5.0
        
        # Content Score (Similarity)
        if m_id in embeddings_dict:
            sim = cosine_similarity(input_vector, embeddings_dict[m_id])
            content_score = (sim + 1) / 2 # Normalize ke 0-1
        else:
            content_score = 0.5
            
        hybrid_score = (alpha * svd_score) + ((1 - alpha) * content_score)
        
        results.append({
            "title": row['title'],
            "hybrid_score": hybrid_score,
            "svd_score": svd_score,
            "content_score": content_score
        })
        
    return pd.DataFrame(results).sort_values("hybrid_score", ascending=False).head(top_n)