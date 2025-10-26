-- create the schemas that are now introduced
CREATE SCHEMA IF NOT EXISTS mlops;
CREATE SCHEMA IF NOT EXISTS ai_datasets;
CREATE SCHEMA IF NOT EXISTS model_outputs;

-- Model: defines the business problem
CREATE TABLE IF NOT EXISTS mlops.model (
  model_id        TEXT PRIMARY KEY,
  name            TEXT,               -- 'churn_classifier'
  business_problem TEXT,              -- 'Predict customer churn'
  ticket_number     TEXT,
  owner           TEXT,
  created_at      TIMESTAMP DEFAULT now()
);

-- Dataset snapshot: versioned training data
CREATE TABLE IF NOT EXISTS mlops.dataset (
  dataset_id      TEXT PRIMARY KEY,
  model_id        TEXT REFERENCES mlops.model(model_id),
  data_version    TEXT,               -- '2025-10-22'
  gcs_path        TEXT,               -- gs://.../training/churn/dt=...
  row_count       INTEGER,
  feature_count   INTEGER,
  created_at      TIMESTAMP DEFAULT now()
);

-- Training run: one experiment attempt
CREATE TABLE IF NOT EXISTS mlops.training_run (
  run_id          TEXT PRIMARY KEY,
  model_id        TEXT REFERENCES mlops.model(model_id),
  dataset_id      TEXT REFERENCES mlops.dataset(dataset_id),
  params    TEXT,               -- '{"C":1.0,"max_iter":1000}' as JSON
  metrics    TEXT,               -- '{"auc":0.81,"f1":0.66}'
  artifact   TEXT,               -- gs://.../models/churn/run_2025_10_22_1/
  status          TEXT,               -- 'success'|'failed'
  created_at      TIMESTAMP DEFAULT now()
);

-- Model artifact / version registry
CREATE TABLE IF NOT EXISTS mlops.model_version (
  model_version_id TEXT PRIMARY KEY,
  model_id         TEXT REFERENCES mlops.model(model_id),
  training_run_id  TEXT REFERENCES mlops.training_run(run_id),
  artifact_path    TEXT,
  metrics_json     TEXT,
  status           TEXT,              -- 'candidate'|'approved'|'archived'
  created_at       TIMESTAMP DEFAULT now()
);

-- Deployment (inference)
CREATE TABLE IF NOT EXISTS mlops.deployment (
  deployment_id    TEXT PRIMARY KEY,
  model_version_id TEXT REFERENCES mlops.model_version(model_version_id),
  endpoint_url     TEXT,              -- Cloud Function URL
  traffic_split    REAL,              -- 1.0 = 100% traffic
  deployed_at      TIMESTAMP DEFAULT now()
);
