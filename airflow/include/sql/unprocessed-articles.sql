SELECT 
    a.id,
    a.headline,
    a.story,
    a.published,
    a.game_id
FROM nfl.stage.dim_articles a
LEFT JOIN nfl.stage.article_embeddings p 
    ON a.id = p.article_id
WHERE p.article_id IS NULL;