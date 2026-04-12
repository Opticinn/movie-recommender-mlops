# scripts/simulate_traffic.py
# Generate dummy prediction logs untuk testing drift detection.
# Jalankan sekali untuk populate tabel prediction_logs.

import sys
import os
import random
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.db_client import get_client, insert_prediction_log

# ── Konfigurasi simulasi ───────────────────────────────────
TOTAL_SESSIONS  = 50    # Jumlah sesi user yang disimulasikan
REQUESTS_PER_SESSION = 3  # Rata-rata request per sesi
DELAY_BETWEEN_LOGS = 0.1  # Jeda antar insert (detik)

# Variasi latency yang realistis (dalam ms)
LATENCY_NORMAL  = (50, 300)    # Range latency normal
LATENCY_SLOW    = (1500, 3000) # Range latency tinggi (simulasi drift)
SLOW_PROBABILITY = 0.1         # 10% chance latency tinggi

# Variasi api_source
API_SOURCES = ["tmdb", "tmdb", "tmdb", "omdb", "fallback"]
# tmdb lebih sering muncul — realistis


# ── Load film dari database ────────────────────────────────
def load_movies_from_db() -> list:
    """
    Ambil semua film dari cache_ratings sebagai list of dict.
    """
    try:
        client = get_client()
        response = client.table("cache_ratings").select(
            "movie_id, title, genre_names"
        ).execute()

        movies = response.data
        print(f"✅ Loaded {len(movies)} movies from database")
        return movies

    except Exception as e:
        print(f"❌ Failed to load movies: {e}")
        return []


# ── Generate satu sesi simulasi ────────────────────────────
def simulate_session(movies: list, session_id: str) -> int:
    """
    Simulasikan satu sesi user — beberapa request rekomendasi.
    Return jumlah request yang berhasil di-log.
    """
    success_count = 0
    num_requests = random.randint(1, REQUESTS_PER_SESSION * 2)

    for _ in range(num_requests):
        # Pilih film input secara acak
        input_movie = random.choice(movies)

        # Pilih 10 film rekomendasi acak (exclude input)
        other_movies = [m for m in movies if m["movie_id"] != input_movie["movie_id"]]
        recommended = random.sample(other_movies, min(10, len(other_movies)))

        # Generate latency yang realistis
        if random.random() < SLOW_PROBABILITY:
            latency_ms = random.randint(*LATENCY_SLOW)   # simulasi lambat
        else:
            latency_ms = random.randint(*LATENCY_NORMAL) # normal

        # Kumpulkan genre rekomendasi
        recommended_genres = []
        for movie in recommended:
            recommended_genres.extend(movie.get("genre_names") or [])

        # Insert log
        ok = insert_prediction_log(
            session_id=session_id,
            input_movie_id=input_movie["movie_id"],
            input_movie_title=input_movie["title"],
            input_movie_genres=input_movie.get("genre_names") or [],
            recommended_movies=[m["title"] for m in recommended],
            recommended_genres=list(set(recommended_genres)),
            latency_ms=latency_ms,
            api_source=random.choice(API_SOURCES),
            user_type="dummy"
        )

        if ok:
            success_count += 1

        time.sleep(DELAY_BETWEEN_LOGS)

    return success_count


# ── Loop utama ─────────────────────────────────────────────
def run_simulation():
    print("🤖 Starting traffic simulation...\n")

    # Load data film
    movies = load_movies_from_db()
    if not movies:
        print("❌ No movies found. Run fetch_ratings.py first.")
        return

    total_success = 0
    total_requests = 0

    for i in range(1, TOTAL_SESSIONS + 1):
        # Generate session ID unik per sesi
        session_id = f"dummy-session-{i:03d}-{random.randint(1000, 9999)}"

        print(f"👤 Session {i}/{TOTAL_SESSIONS} → {session_id}")

        success = simulate_session(movies, session_id)
        total_success += success
        total_requests += success

    # ── Laporan akhir ──────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"✅ Total logs inserted : {total_success}")
    print(f"👤 Total sessions      : {TOTAL_SESSIONS}")
    print(f"{'='*40}")
    print("🎉 Simulation complete!")


if __name__ == "__main__":
    run_simulation()