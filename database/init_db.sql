-- ============================================
-- FILE: init_db.sql
-- Jalankan SEKALI saat setup awal project
-- ============================================

-- Aktifkan extension UUID (wajib untuk gen_random_uuid)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS baseline_metrics (
    metric_name   TEXT        PRIMARY KEY,
    metric_value  NUMERIC     NOT NULL,
    metadata      JSONB       DEFAULT '{}',
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cache_ratings (
    movie_id     INTEGER      PRIMARY KEY,
    title        TEXT         NOT NULL,
    rating       NUMERIC(3,2) CHECK (rating >= 0 AND rating <= 10),
    vote_count   INTEGER      DEFAULT 0,
    release_year INTEGER,
    genre_ids    INTEGER[],
    genre_names  TEXT[],
    last_updated TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_history (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    drift_type  TEXT        NOT NULL CHECK (drift_type IN ('rating', 'genre', 'latency', 'api')),
    severity    TEXT        NOT NULL CHECK (severity IN ('green', 'yellow', 'red')),
    details     JSONB       DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp            TIMESTAMPTZ DEFAULT NOW(),
    session_id           TEXT        NOT NULL,
    user_type            TEXT        DEFAULT 'dummy' CHECK (user_type IN ('dummy', 'real')),
    input_movie_id       INTEGER     NOT NULL,
    input_movie_title    TEXT        NOT NULL,
    input_movie_genres   TEXT[],
    recommended_movies   TEXT[],
    recommended_genres   TEXT[],
    model_version        TEXT        DEFAULT 'v1.0',
    latency_ms           INTEGER     CHECK (latency_ms >= 0),
    api_source           TEXT        CHECK (api_source IN ('tmdb', 'omdb', 'fallback')),
    error_message        TEXT
);