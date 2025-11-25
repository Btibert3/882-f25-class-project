CREATE TABLE IF NOT EXISTS nfl.stage.article_embeddings (
    article_id INTEGER PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);