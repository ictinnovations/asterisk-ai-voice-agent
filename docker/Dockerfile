FROM python:3.11-slim

# espeak-ng is required by Piper's phonemizer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sidecar/ ./sidecar/
COPY download_voices.sh ./

# Config + personas + voices are mounted at runtime (see docker-compose.yml).
ENV AI_AGENT_CONFIG=/app/config.yaml \
    AI_AGENT_PERSONAS=/app/personas.yaml \
    AI_AGENT_LOG=INFO

EXPOSE 9091 9092
WORKDIR /app/sidecar
CMD ["python", "agent.py"]
