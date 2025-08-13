set -e
python -m src.data.parse_camels_raw
python -m src.data.build_daily_table
python -m src.data.to_parquet
