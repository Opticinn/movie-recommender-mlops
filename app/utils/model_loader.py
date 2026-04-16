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
    alpha: float = 0.3, # Kita buat 50:50 agar lebih seimbang
) -> pd.DataFrame:
    
    # 1. Pastikan film input ada di data SBERT
    if input_movie_id not in embeddings_dict:
        return pd.DataFrame()
    input_vector = embeddings_dict[input_movie_id]
    
    # 2. Ambil data Latent Vector Item (qi) dari SVD
    qi_data = svd_model.get('qi', {})
    
    # Fungsi bantu yang aman untuk mengambil vector film (support Dict & Numpy)
    def safe_get_qi(item_id):
        if isinstance(qi_data, dict):
            return qi_data.get(item_id)
        if isinstance(qi_data, (np.ndarray, list)):
            try:
                return qi_data[int(item_id)]
            except (IndexError, ValueError, TypeError):
                return None
        return None

    # Ambil pola rating dari film input
    qi_input = safe_get_qi(input_movie_id)

    results = []
    for _, row in candidate_movies.iterrows():
        m_id = int(row['movie_id'])
        
        # Jangan rekomendasikan film yang sedang dicari
        if m_id == input_movie_id:
            continue
            
        # --- 3. Hitung Content Score (Kemiripan Sinopsis via SBERT) ---
        if m_id in embeddings_dict:
            sim_content = cosine_similarity(input_vector, embeddings_dict[m_id])
            content_score = (sim_content + 1) / 2 # Normalize dari [-1, 1] ke [0, 1]
        else:
            content_score = 0.0
            
        # --- 4. Hitung Collaborative Score (Kemiripan Pola Rating via SVD) ---
        qi_candidate = safe_get_qi(m_id)
        
        if qi_input is not None and qi_candidate is not None:
            # Hitung cosine similarity antar Latent Vector film
            norm_input = np.linalg.norm(qi_input)
            norm_cand = np.linalg.norm(qi_candidate)
            
            if norm_input > 0 and norm_cand > 0:
                sim_collab = np.dot(qi_input, qi_candidate) / (norm_input * norm_cand)
                svd_score = (sim_collab + 1) / 2 # Normalize
            else:
                svd_score = 0.0
        else:
            svd_score = 0.0 # Jika film tidak dikenali SVD, beri nilai 0
            
        # --- 5. Gabungkan menjadi Hybrid Score ---
        hybrid_score = (alpha * svd_score) + ((1 - alpha) * content_score)
        
        # Salin semua data film dan tambahkan skor
        movie_data = row.to_dict()
        movie_data.update({
            "hybrid_score": hybrid_score,
            "svd_score": svd_score,
            "content_score": content_score
        })
        results.append(movie_data)
        
    # Urutkan berdasarkan nilai hybrid tertinggi
    return pd.DataFrame(results).sort_values("hybrid_score", ascending=False).head(top_n)