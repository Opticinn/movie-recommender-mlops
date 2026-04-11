# scripts/fetch_ratings.py
# Fetch 500 film populer dari TMDB dan simpan ke cache_ratings.
# Dijalankan harian via GitHub Actions.

import sys
import os
import time
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.db_client import upsert_cache_rating

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
TOTAL_PAGES = 25      # 25 halaman × 20 film = 500 film
DELAY_SECONDS = 0.25  # Jeda antar request — hormati rate limit TMDB


def fetch_popular_movies(page: int) -> list:
    """
    Fetch satu halaman film populer dari TMDB.
    Return list of movie data, atau [] kalau gagal.
    """
    url = f"{BASE_URL}/movie/popular"
    params = {
        "api_key": TMDB_API_KEY,
        "page": page,
        "language": "en-US"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise error kalau status bukan 200
        return response.json().get("results", [])

    except requests.exceptions.Timeout:
        print(f"   ⚠️  Timeout di halaman {page} — skip")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"   ❌ HTTP Error di halaman {page}: {e}")
        return []
    except Exception as e:
        print(f"   ❌ Error di halaman {page}: {e}")
        return []
    

# Mapping genre_id → genre_name dari TMDB
# Hardcode di sini agar tidak perlu hit API tambahan setiap film
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation",
    35: "Comedy", 80: "Crime", 99: "Documentary",
    18: "Drama", 10751: "Family", 14: "Fantasy",
    36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}


def parse_movie(movie: dict) -> dict | None:
    """
    Konversi raw data dari TMDB → format yang siap disimpan ke database.
    Return None kalau data tidak lengkap.
    """
    # Validasi field wajib ada
    if not movie.get("id") or not movie.get("title"):
        return None

    # Ambil release_year dari release_date (format: "2024-03-15")
    release_date = movie.get("release_date", "")
    release_year = int(release_date[:4]) if release_date else None

    # Konversi genre_ids → genre_names pakai GENRE_MAP
    genre_ids = movie.get("genre_ids", [])
    genre_names = [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]

    return {
        "movie_id": movie["id"],
        "title": movie["title"],
        "rating": round(movie.get("vote_average", 0), 2),
        "vote_count": movie.get("vote_count", 0),
        "release_year": release_year,
        "genre_ids": genre_ids,
        "genre_names": genre_names
    }
    
def fetch_and_store():
    """
    Loop utama: fetch 500 film dari TMDB dan simpan ke cache_ratings.
    """
    print("🎬 Starting TMDB fetch...\n")

    total_success = 0
    total_skip = 0
    total_fail = 0

    for page in range(1, TOTAL_PAGES + 1):
        print(f"📄 Halaman {page}/{TOTAL_PAGES}...")
        movies = fetch_popular_movies(page)

        if not movies:
            print(f"   ⚠️  Tidak ada data di halaman {page}, skip")
            continue

        for movie in movies:
            # Konversi raw data → format database
            parsed = parse_movie(movie)

            if parsed is None:
                total_skip += 1
                continue

            # Simpan ke database
            ok = upsert_cache_rating(**parsed)

            if ok:
                total_success += 1
            else:
                total_fail += 1

        # Jeda antar halaman — hormati rate limit TMDB
        time.sleep(DELAY_SECONDS)

    # Laporan akhir
    print(f"\n{'='*40}")
    print(f"✅ Berhasil disimpan : {total_success} film")
    print(f"⚠️  Di-skip          : {total_skip} film")
    print(f"❌ Gagal             : {total_fail} film")
    print(f"{'='*40}")
    print("🎉 Fetch selesai!")


if __name__ == "__main__":
    fetch_and_store()