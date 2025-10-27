INSERT INTO nfl.mlops.dataset (
  dataset_id,
  model_id,
  data_version,
  gcs_path,
  row_count,
  feature_count,
  created_at
)
VALUES (
  '{{ dataset_id }}',
  '{{ model_id }}',
  '{{ data_version }}',
  'nfl.ai_datasets.player_fantasy_features',
  {{ row_count }},
  {{ feature_count }},
  NOW()
)
ON CONFLICT (dataset_id) DO UPDATE SET
  row_count = EXCLUDED.row_count,
  feature_count = EXCLUDED.feature_count,
  created_at = NOW();