CAMELS-CL Hydrology ML — Hybrid Daily Discharge Prediction
==========================================================

Predict daily streamflow (discharge) for Chilean basins using meteorological forcings, snow, and basin attributes. The project implements a reproducible, memory-safe pipeline and a hybrid modeling strategy:

- Per-basin models for “large” basins.
- Global fallback with Target Encoding on basin_id.
- Diagnostics and serving policy to gate between local/global.
- FastAPI inference locally and AWS Lambda (container image) for serverless production.

Dataset: CAMELS-CL (https://camels.cr2.cl/) (not redistributed here). You must provide raw data locally.

----------------------------------------------------------------
Requirements
------------
  - Python 3.11
  - Docker (with buildx)
  - AWS CLI v2 configured (profile camelscl)
  - Git
  - PowerShell 7+ (Windows) or bash/zsh (Linux/Mac)

----------------------------------------------------------------

Highlights
------------
  - End-to-end pipeline: ingestion → features → training → diagnostics → serving.
  - Memory-safe global training (streamed construction, capped sampling).
  - Feature core via hydrologically-motivated regex filters to reduce dimensionality.
  - Target Encoding (TE) of basin_id for the global model.
  - Near-zero-variance & NaN policy + persistence NSE checks.
  - Re-train “outliers” with local-lite models using top global features.
  - Deterministic serving: persisted feature order, TE mapping, and serving policy.
  - Production-grade serving: FastAPI, Docker, AWS Lambda (Function URL with IAM).
  - Observability: health/ready/metadata endpoints; CloudWatch/X-Ray; budgets & alarms.
----------------------------------------------------------------

Repository Layout
------------
<pre> ```text camelscl-hydrology-ml/ ├─ data/ │ ├─ raw/ # CAMELS-CL original files (not included) │ └─ processed/ │ ├─ *.parquet # normalized tables │ ├─ master_parts/ # master_<basin>.parquet per basin │ └─ features_parts/ # features_<basin>.parquet per basin ├─ models/ # training artifacts (dev runs) ├─ models_release/ # frozen artifacts for serving (prod-ready) │ ├─ xgb_global.json │ ├─ global_basin_te.json # {"mapping": {...}, "global_mean": ...} │ ├─ global_feature_order.json # {"feature_order": [..., "basin_te"]} │ ├─ basin_models_index.json # per-basin model paths (optional) │ └─ serving_policy.json # final gating policy ├─ src/ │ ├─ preprocessing/ │ │ ├─ parse_attributes_transposed.py │ │ ├─ parse_timeseries_wide.py │ │ └─ build_master_partitioned.py │ ├─ features/ │ │ └─ build_features.py │ ├─ models/ │ │ ├─ train_xgboost_cpu.py # simple baseline (optional) │ │ ├─ train_hybrid_models.py # main hybrid trainer (per-basin + global) │ │ ├─ diagnose_basins.py # quality & metrics diagnostics │ │ ├─ build_policy_from_diagnostics.py │ │ ├─ augment_metrics_with_persistence.py │ │ └─ retrain_outliers.py # local-lite retraining for worst basins │ └─ serving/ │ ├─ inference_handler.py # FastAPI app + Lambda handler │ └─ __init__.py ├─ Dockerfile # local serving image (FastAPI on 0.0.0.0:8080) ├─ Dockerfile.lambda # AWS Lambda container image ├─ requirements.txt / requirements-serving.txt └─ README.md ``` </pre>

All Python packages under src/ include __init__.py to ensure imports work locally and inside Lambda.

----------------------------------------------------------------
Quickstart (Local)

  - Linux/Mac

```bash
export MODELS_DIR=./models_release
uvicorn src.serving.inference_handler:app --host 0.0.0.0 --port 8000 --reload
```
  - Windows (PowerShell)

```powershell
$env:MODELS_DIR = ".\models_release"
uvicorn src.serving.inference_handler:app --host 0.0.0.0 --port 8000 --reload
```
----------------------------------------------------------------

Data & Features
---------------
Raw → Processed
- parse_attributes_transposed.py: robust TSV parsing (quotes, comments), numeric coercion → data/processed/attributes.parquet.
- parse_timeseries_wide.py: wide to long for daily series → data/processed/<variable>.parquet.
- build_master_partitioned.py: join all sources per basin_id and date → one Parquet per basin in master_parts/.

Features
- build_features.py: lags (1/2/3/7/14/30), rolling sums/means, degree-days, hydrologic deficit, SWE lags.
- Target: y_log1p = log1p(discharge_mm) (stable variance).
- Static attributes appended to each basin time series.
- Output: features_parts/features_<basin>.parquet.

----------------------------------------------------------------

Modeling
--------
1) Hybrid Trainer (src/models/train_hybrid_models.py)
- Per-basin models (XGBoost) for basins with enough rows (MIN_ROWS_BASIN).
  - Imputation rules: NaNs in lags/rolling/deficit/SWE → 0; others → train median.
- Global model (XGBoost) with Target Encoding (basin_te):
  - Pass 1: compute TE from train windows across basins (sum/count streamed).
  - Pass 2: build global train/test with caps per basin (avoid OOM), select core features by regex and intersect across basins, apply TE and imputation.
  - Persist exact feature order (global_feature_order.json) including "basin_te" last.
- Artifacts:
  - models/xgb_global.json
  - models/global_basin_te.json
  - models/global_feature_order.json
  - models/basin_models_index.json
  - models/metrics_hybrid.json
  - models/metrics_per_basin.csv

Run:
  python -m src.models.train_hybrid_models

2) Diagnostics & Policy
- diagnose_basins.py: loads feature files per basin, computes metrics, missingness, and stability indicators.
  Outputs:
    - models/diagnostics/summary.csv
    - models/diagnostics/missing_top15/<basin>.csv
    - models/diagnostics/near_zero_report.csv
- build_policy_from_diagnostics.py: generates a gating policy (exclusions per basin for high-NaN columns, near-zero-variance flags, rmse mode tags, etc.).
  Output: models/diagnostics/gating_policy.json
- augment_metrics_with_persistence.py: compute persistence NSE (naive “y[t-1]” baseline) on the test split and append to metrics.
  Output: models/metrics_per_basin_with_persistence.csv
- retrain_outliers.py: select worst basins (bottom-N by R²), train local-lite models with top-K global feature importances (excluding basin_te), and compare against original per-basin and persistence.
  Outputs:
    - models/retrained/basins_local_lite/xgb_basin_<id>_lite.json
    - models/retrained/retrained_metrics.csv
    - models/retrained/retrained_compare.csv

Gating Policy (serving)
- Final serving_policy.json merges:
  - If retrained_compare.suggested_serving == "local_lite" and ΔR² > 0 → use local_lite.
  - If metrics_per_basin_with_persistence.suggested_serving == "global" → use global.
  - If a strong per-basin full model exists, keep local (optional).
- Schema:
  {
    "basins": {
      "3434003": {
        "mode": "local_lite",
        "model_path": "retrained/basins_local_lite/xgb_basin_3434003_lite.json",
        "feature_order": ["prcp_mm_cr2met_lag1", "tmean_c_lag1", "..."]
      },
      "7317003": { "mode": "global" },
      "2120001": { "mode": "local" }
    }
  }

----------------------------------------------------------------

Serving (Local)
---------------
FastAPI app at src/serving/inference_handler.py:
- Endpoints:
  - GET /healthz – static presence of artifacts.
  - GET /readyz – lazy load global model/order/TE to ensure readiness.
  - GET /metadata – file hashes, sizes, mtime for change audit.
  - POST /predict – inference.
- Deterministic feature vector:
  - Global path uses global_feature_order.json + basin_te from global_basin_te.json.
  - Local-lite path uses per-model feature_order embedded in serving policy.
- Imputation at inference:
  - Missing lags/rolling/deficit/SWE → 0.
  - Others → 0 (production option: versioned medians).

Run locally:
  export MODELS_DIR=./models_release
  uvicorn src.serving.inference_handler:app --host 0.0.0.0 --port 8000 --reload

Predict example:
  curl -s http://127.0.0.1:8000/readyz
  curl -s -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "basin_id": "3434003",
      "features": {
        "prcp_mm_cr2met_lag1": 2.1,
        "pet_mm_hargreaves_lag3": 1.0,
        "deficit_sum_7d": 0.0,
        "tmean_c_lag1": 12.3,
        "swe_mm_lag1": 4.5
      }
    }'

----------------------------------------------------------------

Docker (Local)
--------------
FastAPI (dev):
  docker build -t camelscl-serving:latest .
  docker run --rm -p 8080:8080 camelscl-serving:latest
  curl -s http://localhost:8080/healthz

Image contents:
- /app/src application code
- /app/models frozen artifacts (models_release/)
- Uvicorn starts on :8080.

----------------------------------------------------------------

AWS Lambda (Container Image)
----------------------------
Keep it simple: one function with a Function URL protected by AWS_IAM.

Build & Push:
- Base image: public.ecr.aws/lambda/python:3.11
- Dockerfile: Dockerfile.lambda
- Handler: src.serving.inference_handler.handler (via Mangum)

ECR login:
  aws ecr get-login-password --region <REGION> \
  | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com

Create repository (once):
  aws ecr create-repository --repository-name camelscl-serving --region <REGION> || true

Build (linux/amd64), tag and push:
  docker buildx build --platform linux/amd64 -f Dockerfile.lambda -t camelscl-serving:v1 .
  docker tag camelscl-serving:v1 <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/camelscl-serving:v1
  docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/camelscl-serving:v1

Create / Update Lambda:
  aws lambda create-function \
    --function-name camelscl-inference \
    --package-type Image \
    --code ImageUri=<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/camelscl-serving:v1 \
    --role arn:aws:iam::<ACCOUNT>:role/<LambdaExecRole> \
    --timeout 10 --memory-size 1024 \
    --architecture x86_64 \
    --environment "Variables={MODELS_DIR=/var/task/models,LOG_LEVEL=INFO}" \
    --region <REGION>

  aws lambda create-function-url-config \
    --function-name camelscl-inference \
    --auth-type AWS_IAM \
    --cors "AllowOrigins=['*'],AllowHeaders=['*'],AllowMethods=['*']" \
    --region <REGION>

  aws lambda update-function-code \
    --function-name camelscl-inference \
    --image-uri <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/camelscl-serving:v1 \
    --region <REGION>

Smoke tests:
- GET {FnUrl}healthz
- GET {FnUrl}readyz
- POST {FnUrl}predict (SigV4 signed if using IAM).

----------------------------------------------------------------
Signed invocation (IAM + SigV4)

Since the Function URL is protected by AWS IAM, use the provided client:

Environment variables

- `AWS_PROFILE=camelsc1`
- `AWS_REGION=sa-east-1`
- `FN_URL=<your Function URL, must end with '/'>`
Run

```powershell
python .\signed_post.py
```

> **Note:** Do not hardcode `FN_URL` or credentials in code. Use environment variables.
> The 12-digit AWS Account ID is **not** secret, but your Function URL should be treated as sensitive.
----------------------------------------------------------------
PowerShell deploy script

A convenience script to build → push → update Lambda → smoke test:

```powershell
.\deploy.ps1 `
  -Region "sa-east-1" `
  -Profile "camelsc1" `
  -RepoName "camelsc1-serving" `
  -FnName "camelsc1-inference" `
  -Tag "v1"
```

The script logs into ECR, builds & pushes the image, updates the Lambda, and runs a smoke test with `signed_post.py`.
----------------------------------------------------------------

Reproducibility & Release
-------------------------
- Freeze serving artifacts under models_release/:
  - xgb_global.json, global_basin_te.json, global_feature_order.json, serving_policy.json
- The inference code trusts exact feature order and TE mapping to avoid training/serving drift.
- Version images by tag (v1, v2, …) and/or digest in Lambda.

----------------------------------------------------------------

Observability & Cost Guardrails (AWS)
-------------------------------------
- CloudWatch Logs: /aws/lambda/camelscl-inference, retention 7–30 days.
- X-Ray: tracing Active (optional).
- CloudWatch Alarms: 5xx errors, throttles, p95 duration, invocation spikes.
- Budgets: monthly cost caps with email alerts (and optional anomaly detection).
- Concurrency cap (e.g., reserved-concurrent-executions=10) to prevent runaway costs.

With a Function URL protected by IAM and no inbound traffic, Lambda costs are typically near zero. ECR charges minimal storage.

----------------------------------------------------------------

Practical Notes & Gotchas
-------------------------
- Windows paths: prefer raw strings (r"C:\...") or forward slashes. Avoid unicodeescape errors.
- JSON payloads: use double quotes and dot decimals. In PowerShell, pipe a valid JSON string to file and use --data-binary "@file.json".
- Dockerfile on PowerShell: don’t paste Here-Strings into Dockerfile (no @' ... '@), just plain text.
- XGBoost CPU: libgomp1 required in slim images. Lambda container already includes what’s needed when using Python base, but install in custom images.
- NumPy shape mismatches (global vstack): fixed by enforcing a fixed global feature order and filtering per-basin matrices to that exact order.
- Type checkers (Pylance): use explicit List[str], Dict[str, float] where needed; cast to float(np.asarray(...).ravel()[0]) for predictions.

----------------------------------------------------------------

How to Reproduce (Minimal)
--------------------------
# 1) Preprocess & features
python -m src.preprocessing.parse_attributes_transposed
python -m src.preprocessing.parse_timeseries_wide
python -m src.preprocessing.build_master_partitioned
python -m src.features.build_features

# 2) Train hybrid (per-basin + global)
python -m src.models.train_hybrid_models

# 3) Diagnostics & policy
python -m src.models.diagnose_basins --data_dir data/processed/features_parts --metrics_csv models/metrics_per_basin.csv --split_date 2009-01-01 --out_dir models/diagnostics
python -m src.models.augment_metrics_with_persistence --data_dir data/processed/features_parts --metrics_csv models/metrics_per_basin.csv --split_date 2009-01-01 --out_csv models/metrics_per_basin_with_persistence.csv
python -m src.models.retrain_outliers --data_dir data/processed/features_parts --metrics_csv models/metrics_per_basin.csv --global_model models/xgb_global.json --global_feature_order models/global_feature_order.json --policy_json models/diagnostics/gating_policy.json --split_date 2009-01-01 --n_bottom 20 --top_k 60 --out_dir models/retrained

# 4) Freeze artifacts for serving
# Copy/curate to models_release/: xgb_global.json, global_basin_te.json, global_feature_order.json, serving_policy.json

# 5) Serve locally (FastAPI)
export MODELS_DIR=./models_release
uvicorn src.serving.inference_handler:app --host 0.0.0.0 --port 8000 --reload

----------------------------------------------------------------

API (Serving)
-------------
GET /healthz
  Static presence check of required artifacts.

GET /readyz
  Lazy load global model, TE, and order to confirm readiness.

GET /metadata
  Returns hashes, sizes, mtimes of artifacts (helps validate the running revision).

POST /predict
  Body:
  {
    "basin_id": "3434003",
    "features": {
      "prcp_mm_cr2met_lag1": 2.1,
      "pet_mm_hargreaves_lag3": 1.0,
      "deficit_sum_7d": 0.0,
      "tmean_c_lag1": 12.3,
      "swe_mm_lag1": 4.5
    }
  }

  Response:
  {
    "basin_id": "3434003",
    "yhat_log": 0.7172,
    "yhat_mm": 1.0489,
    "model_used": "global",
    "mode": "global",
    "features_missing": [ "..."],
    "basin_te": 0.0305,
    "notes": []
  }

----------------------------------------------------------------

Troubleshooting

- **404 on POST to Function URL** → ensure you include `/predict` (`POST {fnUrl}predict`).

- `Runtime.HandlerNotFound` in Lambda → check `CMD ["src.serving.inference_handler.handler"]` and that both `src/` and `src/serving/` have `__init__.py`.

- **"Invalid JSON" when creating CORS via CLI** → on Windows, use a file (`cors.json`) or escape quotes correctly.

- **CloudWatch Logs "tail" on Windows** → use `describe-log-streams` to get the latest stream, then `get-log-events`.

- **CRLF/LF differences on Windows** → use `.gitattributes` to normalize line endings.
----------------------------------------------------------------

License & Data
--------------
- Code: MIT (adjust to your needs).
- Data: CAMELS-CL is owned/maintained externally. You must comply with their license and terms. This repo does not ship the dataset.

Acknowledgments
---------------
- CAMELS-CL dataset & community.
- XGBoost, FastAPI, Mangum, AWS Lambda container runtime.

Roadmap (Optional)
------------------
- Quantile modeling for predictive intervals.
- Add LightGBM and CatBoost baselines.
- Store per-basin train medians to improve online imputation.
- Batch inference & backfilling (Step Functions + S3).
- GitHub Actions for CI/CD (build image, update Lambda, smoke tests).
