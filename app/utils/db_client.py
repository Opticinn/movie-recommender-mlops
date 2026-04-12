# app/utils/db_client.py
# Wrapper untuk semua operasi database Supabase

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from supabase import create_client, Client
import requests as req

# Load .env sekali saat module ini diimport
load_dotenv()

def get_client() -> Client:
    """
    Membuat dan mengembalikan Supabase client.
    Dipanggil setiap kali file lain butuh koneksi database.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("❌ SUPABASE_URL atau SUPABASE_ANON_KEY tidak ditemukan!")

    return create_client(url, key)

def insert_prediction_log(
    session_id: str,
    input_movie_id: int,
    input_movie_title: str,
    recommended_movies: list,
    latency_ms: int,
    input_movie_genres: list = None,
    recommended_genres: list = None,
    api_source: str = "tmdb",
    user_type: str = "dummy",
    error_message: str = None,
    model_version: str = "v1.0"
) -> bool:
    """
    Menyimpan satu record log rekomendasi ke tabel prediction_logs.
    Return True jika berhasil, False jika gagal.
    """
    try:
        client = get_client()

        data = {
            "session_id": session_id,
            "user_type": user_type,
            "input_movie_id": input_movie_id,
            "input_movie_title": input_movie_title,
            "input_movie_genres": input_movie_genres or [],
            "recommended_movies": recommended_movies,
            "recommended_genres": recommended_genres or [],
            "model_version": model_version,
            "latency_ms": latency_ms,
            "api_source": api_source,
            "error_message": error_message
        }

        client.table("prediction_logs").insert(data).execute()
        return True

    except Exception as e:
        print(f"❌ Gagal insert prediction log: {e}")
        return False
    

def upsert_cache_rating(
    movie_id: int,
    title: str,
    rating: float,
    vote_count: int,
    release_year: int,
    genre_ids: list = None,
    genre_names: list = None,
    poster_path: str = None, 
) -> bool:
    """
    Menyimpan data film ke cache_ratings.
    Jika movie_id sudah ada → UPDATE, belum ada → INSERT.
    """
    try:
        client = get_client()

        data = {
            "movie_id": movie_id,
            "title": title,
            "rating": rating,
            "vote_count": vote_count,
            "release_year": release_year,
            "genre_ids": genre_ids or [],
            "genre_names": genre_names or [],
            "poster_path": poster_path
        }

        client.table("cache_ratings").upsert(
            data,
            on_conflict="movie_id"  # kalau movie_id sama → UPDATE
        ).execute()
        return True

    except Exception as e:
        print(f"❌ Gagal upsert cache rating: {e}")
        return False
    
    
def get_baseline_metrics() -> dict:
    """
    Mengambil semua baseline metrics dari database.
    Return dictionary {metric_name: metric_value}.
    """
    try:
        client = get_client()

        response = client.table("baseline_metrics").select("*").execute()

        # Ubah dari list of rows → dictionary yang lebih mudah dipakai
        # Dari: [{"metric_name": "mean_rating", "metric_value": 3.5}, ...]
        # Ke:   {"mean_rating": 3.5, ...}
        metrics = {
            row["metric_name"]: row["metric_value"]
            for row in response.data
        }

        return metrics

    except Exception as e:
        print(f"❌ Gagal mengambil baseline metrics: {e}")
        return {}
    
# ── Telegram Alert ─────────────────────────────────────────
def send_telegram_alert(message: str) -> bool:
    """
    Kirim notifikasi ke Telegram ketika drift terdeteksi.
    Return True jika berhasil, False jika gagal.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram credentials not found, skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"  # support bold, italic, code
    }

    try:
        response = req.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Telegram alert sent!")
        return True
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")
        return False
    

