# scripts/generate_embeddings.py
# Generate SBERT embeddings untuk semua film di cache_ratings.
# Jalankan sekali sebelum deploy — hasilnya disimpan ke model/sbert_embeddings.pkl

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db_client import get_client
from app.utils.model_loader import get_sbert_embeddings

def main():
    print("🎬 Fetching all movie IDs from cache_ratings...\n")

    # Ambil semua movie_id dari database
    client = get_client()
    response = client.table("cache_ratings").select("movie_id").execute()
    movie_ids = [row["movie_id"] for row in response.data]

    print(f"✅ Found {len(movie_ids)} movies\n")

    # Generate embeddings untuk semua film
    # Force regenerate karena tadi hanya 10 film
    embeddings_dict, _ = get_sbert_embeddings(
        movie_ids=movie_ids,
        force_regenerate=True
    )

    print(f"\n🎉 Done! {len(embeddings_dict)} embeddings saved to model/sbert_embeddings.pkl")


if __name__ == "__main__":
    main()