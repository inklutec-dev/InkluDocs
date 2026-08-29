# InkluDocs Umwandler (29.08.2026): Word -> barrierefreie PDF (PDF/UA) ueber
# LibreOffice, danach Pruefung mit veraPDF. Eigener kleiner Dienst neben der App,
# damit das App-Image klein bleibt und die Umwandlung abgeschottet laeuft
# (Zeitlimit, eigenes Profil je Lauf). Erreichbar nur im Compose-Netz, Port 8090.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 JAVA_TOOL_OPTIONS="-Xmx512m"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer-nogui \
    fonts-dejavu fonts-liberation fonts-noto-core fonts-crosextra-carlito fonts-crosextra-caladea \
    default-jre-headless \
    python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# veraPDF aus dem offiziellen Image (nur die Jars; Java kommt von Debian)
COPY --from=ghcr.io/verapdf/cli:latest /opt/verapdf /opt/verapdf

RUN pip3 install --no-cache-dir --break-system-packages \
    "fastapi>=0.115,<0.116" "uvicorn>=0.30,<0.31" "python-multipart>=0.0.9,<0.1"

COPY app.py /srv/app.py
WORKDIR /srv
RUN mkdir -p /work && useradd -r -m -d /home/konverter konverter && chown -R konverter:konverter /work /home/konverter
USER konverter
EXPOSE 8090
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090", "--log-level", "warning"]
