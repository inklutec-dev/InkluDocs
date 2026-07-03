FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR with German language pack
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/
COPY frontend/ /app/frontend/

# i18n-Vollstaendigkeits-Check: bricht den Build hart ab, wenn ein
# Template-/JS-String in einem der vier .po-Kataloge fehlt
RUN python /app/scripts/check_i18n.py

# Kompiliere gettext-Uebersetzungen (.po -> .mo) waehrend des Builds
RUN pybabel compile -d /app/locales -f || echo "WARN: pybabel compile failed or no locales"

RUN mkdir -p /app/data/uploads /app/data/results

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
