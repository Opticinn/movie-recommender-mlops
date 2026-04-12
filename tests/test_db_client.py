# tests/test_db_client.py
# Auto-test untuk fungsi-fungsi di db_client.py
# Dijalankan otomatis setiap push via GitHub Actions

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.db_client import (
    get_client,
    insert_prediction_log,
    upsert_cache_rating,
    get_baseline_metrics,
    send_telegram_alert
)

# ── Test runner sederhana tanpa library tambahan ───────────
passed = 0
failed = 0


def test(name: str, condition: bool) -> None:
    global passed, failed
    if condition:
        print(f"   ✅ {name}")
        passed += 1
    else:
        print(f"   ❌ {name}")
        failed += 1


# ══════════════════════════════════════════════════════════
print("🧪 Running db_client tests...\n")

# ── Test 1: Koneksi Supabase ───────────────────────────────
print("📡 Test: get_client()")
try:
    client = get_client()
    test("get_client() returns valid client", client is not None)
except Exception as e:
    test(f"get_client() → {e}", False)

# ── Test 2: get_baseline_metrics ──────────────────────────
print("\n📊 Test: get_baseline_metrics()")
try:
    metrics = get_baseline_metrics()
    test("returns a dict", isinstance(metrics, dict))
    test("mean_rating exists", "mean_rating" in metrics)
    test("drift_threshold_rating exists", "drift_threshold_rating" in metrics)
    test("latency_threshold_ms exists", "latency_threshold_ms" in metrics)
except Exception as e:
    test(f"get_baseline_metrics() → {e}", False)

# ── Test 3: upsert_cache_rating ───────────────────────────
print("\n🎬 Test: upsert_cache_rating()")
try:
    ok = upsert_cache_rating(
        movie_id=99999,       # ID dummy — tidak akan konflik
        title="Test Movie",
        rating=7.5,
        vote_count=100,
        release_year=2024,
        genre_ids=[28, 35],
        genre_names=["Action", "Comedy"]
    )
    test("upsert returns True", ok is True)
except Exception as e:
    test(f"upsert_cache_rating() → {e}", False)

# ── Test 4: insert_prediction_log ─────────────────────────
print("\n📋 Test: insert_prediction_log()")
try:
    ok = insert_prediction_log(
        session_id="test-github-actions",
        input_movie_id=99999,
        input_movie_title="Test Movie",
        recommended_movies=["Movie A", "Movie B"],
        latency_ms=50,
        input_movie_genres=["Action"],
        api_source="tmdb",
        user_type="dummy"
    )
    test("insert returns True", ok is True)
except Exception as e:
    test(f"insert_prediction_log() → {e}", False)

# ── Test 5: send_telegram_alert ───────────────────────────
print("\n📱 Test: send_telegram_alert()")
try:
    ok = send_telegram_alert(
        "🧪 <b>Auto-test passed!</b>\n\n"
        "GitHub Actions successfully verified all db_client functions.\n"
        "<code>Movie Recommender MLOps</code>"
    )
    test("alert returns True", ok is True)
except Exception as e:
    test(f"send_telegram_alert() → {e}", False)

# ══════════════════════════════════════════════════════════
print(f"\n{'='*40}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"{'='*40}")

# Exit code 1 kalau ada yang gagal — GitHub Actions akan tandai sebagai FAILED
if failed > 0:
    sys.exit(1)