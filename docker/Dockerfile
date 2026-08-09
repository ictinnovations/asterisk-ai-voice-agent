FROM python:3.11-slim

# espeak-ng is required by Piper's phonemizer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY asterisk_ai_voice_agent/ ./asterisk_ai_voice_agent/
COPY download_voices.sh ./
RUN pip install --no-cache-dir --no-deps .

# Config + personas + voices are mounted at runtime (see docker-compose.yml).
ENV AI_AGENT_CONFIG=/app/config.yaml \
    AI_AGENT_PERSONAS=/app/personas.yaml \
    AI_AGENT_LOG=INFO

EXPOSE 9091 9092
CMD ["asterisk-ai-voice-agent"]
