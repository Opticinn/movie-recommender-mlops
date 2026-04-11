# scripts/insert_baseline.py
# Jalankan SEKALI saat setup awal project.
# Menyimpan nilai baseline ke tabel baseline_metrics.

import sys
import os

# Agar bisa import dari folder app/utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db_client import get_client

# ── 1. Definisi baseline metrics ───────────────────────────
# Nilai dari MovieLens dataset (standar industri untuk recommendation system)

BASELINE_METRICS = [
    {
        "metric_name": "mean_rating",
        "metric_value": 3.5,
        "metadata": {
            "description": "Rata-rata rating film dari MovieLens dataset",
            "source": "MovieLens 25M Dataset",
            "unit": "rating (0-5 scale)"
        }
    },
    {
        "metric_name": "std_rating",
        "metric_value": 1.05,
        "metadata": {
            "description": "Standar deviasi rating — ukuran variasi rating",
            "source": "MovieLens 25M Dataset",
            "unit": "rating"
        }
    },
    {
        "metric_name": "total_films",
        "metric_value": 500,
        "metadata": {
            "description": "Target jumlah film di cache harian",
            "source": "Project design decision",
            "unit": "films"
        }
    },
    {
        "metric_name": "drift_threshold_rating",
        "metric_value": 0.3,
        "metadata": {
            "description": "Batas maksimal pergeseran mean_rating sebelum alert",
            "source": "Project design decision (Fase 1)",
            "unit": "rating points"
        }
    },
    {
        "metric_name": "latency_threshold_ms",
        "metric_value": 2000,
        "metadata": {
            "description": "Batas maksimal latency inferensi sebelum alert",
            "source": "Project design decision",
            "unit": "milliseconds"
        }
    }
]


# ── 2. Insert ke database ───────────────────────────────────

def insert_baseline():
    print("📊 Inserting baseline metrics...\n")

    client = get_client()
    success_count = 0

    for metric in BASELINE_METRICS:
        try:
            client.table("baseline_metrics").upsert(
                metric,
                on_conflict="metric_name"
            ).execute()

            print(f"   ✅ {metric['metric_name']} = {metric['metric_value']}")
            success_count += 1

        except Exception as e:
            print(f"   ❌ {metric['metric_name']} → {e}")

    print(f"\n🎉 {success_count}/{len(BASELINE_METRICS)} metrics inserted successfully!")


# ── 3. Jalankan ────────────────────────────────────────────

if __name__ == "__main__":
    insert_baseline()