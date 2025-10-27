INSERT INTO nfl.mlops.model (
  model_id,
  name,
  business_problem,
  ticket_number,
  owner,
  created_at
)
VALUES (
  '{{ model_id }}',
  '{{ name }}',
  '{{ business_problem }}',
  '{{ ticket_number }}',
  '{{ owner }}',
  NOW()
)
ON CONFLICT (model_id) DO NOTHING;