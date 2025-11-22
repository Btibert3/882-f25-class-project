SELECT 
    i.article_id,
    i.url,
    i.caption,
    i.game_id
FROM nfl.stage.dim_article_images i
LEFT JOIN nfl.stage.article_image_embeddings p 
    ON i.article_id = p.article_id 
    AND i.url = p.image_url
WHERE p.article_id IS NULL