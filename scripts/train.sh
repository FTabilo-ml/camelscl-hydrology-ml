set -e
python -m src.models.train_linear
python -m src.models.evaluate
