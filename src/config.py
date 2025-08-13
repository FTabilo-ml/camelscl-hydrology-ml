from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Columnas mínimas esperadas tras limpieza
FEATURE_COLS = ["prcp_mm", "tmin_c", "tmax_c", "pet_mm"]
TARGET_COL = "discharge_m3s"   # o "discharge_mm"
KEY_COLS = ["basin_id", "date"]  # ISO date
