FROM python:3.11-slim

# xgboost necesita libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# código + artefactos (congelamos models_release dentro de la imagen)
COPY src/ src/
COPY models_release/ models/

# una sola variable que usa tu código
ENV MODELS_DIR=/app/models
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "src.serving.inference_handler:app", "--host", "0.0.0.0", "--port", "8080"]
