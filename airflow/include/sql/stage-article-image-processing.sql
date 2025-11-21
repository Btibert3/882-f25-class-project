CREATE TABLE IF NOT EXISTS nfl.stage.article_image_embeddings (
    article_id INTEGER,
    image_url VARCHAR,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, image_url)
);