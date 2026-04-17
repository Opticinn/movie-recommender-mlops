import os
import sys
import time
import pandas as pd

# Menambahkan root directory ke sys.path agar bisa import dari folder 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db_client import get_client, insert_prediction_log

# --- IMPORT DISESUAIKAN DENGAN MODEL_LOADER KAMU ---
from app.utils.model_loader import get_svd_model, get_sbert_embeddings, get_hybrid_recommendations

def run_simulation():
    print("🤖 Memulai Bot Simulasi Traffic...")
    
    # 1. Load Data
    client = get_client()
    print("📥 Mengambil data film dari database...")
    response = client.table("cache_ratings").select("*").execute()
    movies_df = pd.DataFrame(response.data)
    
    if movies_df.empty:
        print("❌ Data film kosong. Batalkan simulasi.")
        return

    # --- PERBAIKAN: Load Model Menggunakan Fungsi Asli Kamu ---
    print("🧠 Memuat model Hybrid (SVD + SBERT)...")
    try:
        # Load SVD
        svd_model = get_svd_model()
        
        # Load SBERT dan ubah menjadi dictionary
        embeddings, movie_ids = get_sbert_embeddings()
        embeddings_dict = dict(zip(movie_ids.tolist(), embeddings))
        
    except Exception as e:
        print(f"❌ Gagal memuat model. Simulasi dibatalkan. Error: {e}")
        return
    # ------------------------------------------------------------

    # 2. Lakukan 20 pencarian acak per sesi
    TOTAL_REQUESTS = 20
    
    for i in range(TOTAL_REQUESTS):
        try:
            # Pilih film acak seolah-olah user sedang mencari
            random_movie = movies_df.sample(1).iloc[0]
            start_time = time.time()

            # Panggil fungsi rekomendasi dengan parameter lengkap
            recommendations = get_hybrid_recommendations(
                input_movie_id=int(random_movie["movie_id"]),
                candidate_movies=movies_df,
                svd_model=svd_model,              
                embeddings_dict=embeddings_dict   
            )

            # Hitung Latency
            latency = int((time.time() - start_time) * 1000)

            # Insert ke log Supabase (menggunakan model_version V2)
            insert_prediction_log(
                session_id="bot_github_action",
                user_type="bot",
                input_movie_id=int(random_movie["movie_id"]),
                input_movie_title=random_movie["title"],
                input_movie_genres=random_movie["genre_names"],
                recommended_movies=recommendations["title"].tolist() if not recommendations.empty else [],
                latency_ms=latency,
                model_version="v2.0-hybrid"
            )

            print(f"[{i+1}/{TOTAL_REQUESTS}] ✅ Bot mencari: {random_movie['title']} ({latency}ms)")
            
            # Jeda 2 detik agar tidak di-block oleh API Supabase
            time.sleep(2)

        except Exception as e:
            print(f"❌ Error pada simulasi ke-{i+1}: {e}")
            time.sleep(2)

    print("🎉 Sesi simulasi selesai. Bot tidur kembali.")

if __name__ == "__main__":
    run_simulation()