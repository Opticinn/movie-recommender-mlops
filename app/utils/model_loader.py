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
    """Prediksi rating menggunakan model ringan (support Dict & Numpy Array)"""
    
    # 1. Ambil rata-rata global
    mean = model_data.get('mean', model_data.get('mu', model_data.get('global_mean', 0)))
    
    # Fungsi pembantu untuk mengambil data (return None jika tidak ketemu)
    def safe_get(data, key):
        if data is None:
            return None
        if isinstance(data, dict):
            return data.get(key, None)
        if isinstance(data, (np.ndarray, list)):
            try:
                return data[int(key)]
            except (IndexError, ValueError, TypeError):
                return None
        return None

    # 2. Ekstrak struktur data
    bu_data = model_data.get('bu', {})
    bi_data = model_data.get('bi', {})
    pu_data = model_data.get('pu', {})
    qi_data = model_data.get('qi', {})
    
    # 3. Ambil bias (default 0.0 jika None)
    bu = safe_get(bu_data, uid) or 0.0
    bi = safe_get(bi_data, iid) or 0.0
    
    # 4. Ambil vektor latent
    pu = safe_get(pu_data, uid)
    qi = safe_get(qi_data, iid)
    
    # 5. Hitung dot product hanya jika keduanya ditemukan!
    if pu is not None and qi is not None:
        dot_product = np.dot(pu, qi)
    else:
        dot_product = 0.0
    
    # 6. Hitung estimasi rating
    est = mean + bu + bi + dot_product
    
    # 7. Batasi nilai rating antara 0.5 sampai 5.0
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
    # 2. Lakukan iterasi
    for _, row in candidate_movies.iterrows():
        m_id = int(row['movie_id'])
        
        # Mencegah merekomendasikan film yang sama dengan input
        if m_id == input_movie_id:
            continue
        
        # Collaborative Score (SVD)
        svd_score = predict_svd_light(svd_model, user_id, m_id) / 5.0
        
        # Content Score (Similarity)
        if m_id in embeddings_dict:
            sim = cosine_similarity(input_vector, embeddings_dict[m_id])
            content_score = (sim + 1) / 2 # Normalize ke 0-1
        else:
            content_score = 0.5
            
        hybrid_score = (alpha * svd_score) + ((1 - alpha) * content_score)
        
        # SANGAT PENTING: Salin semua data film asli (termasuk rating, poster_path, dll)
        movie_data = row.to_dict()
        movie_data.update({
            "hybrid_score": hybrid_score,
            "svd_score": svd_score,
            "content_score": content_score
        })
        results.append(movie_data)
        
    # 3. Urutkan dan kembalikan
    return pd.DataFrame(results).sort_values("hybrid_score", ascending=False).head(top_n)