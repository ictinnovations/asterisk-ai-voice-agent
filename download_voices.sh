#!/usr/bin/env bash
# Download Piper voices into ./voices. Usage: ./download_voices.sh en_US-amy-medium [more...]
# Browse voices: https://huggingface.co/rhasspy/piper-voices
set -euo pipefail

DEST="${VOICES_DIR:-./voices}"
mkdir -p "$DEST"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Map a voice name to its HF path: <lang>/<lang_REGION>/<name>/<quality>/<file>
# e.g. en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium.onnx
voice_path() {
  local v="$1"
  local lang_region="${v%%-*}"          # en_US
  local rest="${v#*-}"                   # amy-medium
  local name="${rest%%-*}"               # amy
  local quality="${rest#*-}"             # medium
  local lang="${lang_region%%_*}"        # en
  echo "${lang}/${lang_region}/${name}/${quality}/${v}"
}

for v in "$@"; do
  p="$(voice_path "$v")"
  echo "Downloading $v ..."
  curl -fL "$BASE/$p.onnx"      -o "$DEST/$v.onnx"
  curl -fL "$BASE/$p.onnx.json" -o "$DEST/$v.onnx.json"
  echo "  -> $DEST/$v.onnx"
done
echo "Done."
