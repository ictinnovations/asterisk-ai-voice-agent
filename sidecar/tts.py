"""
TTS: Piper (local, ONNX) + ElevenLabs (cloud).

Synthesizes text to PCM, resamples to 8 kHz slin16 (the AudioSocket format),
and yields 320-byte chunks (one 20 ms frame each) so callers can stream them
straight into AudioSocket AUDIO frames.

Providers (persona.tts_provider):
  piper       - local ONNX voice from your voices dir (see download_voices.sh).
                Default en_US-amy-medium (22.05 kHz). Override via
                config.providers.piper.default_voice or persona.tts_voice_id
                (basename of the .onnx file).
  elevenlabs  - cloud API (needs an ElevenLabs API key and persona.tts_voice_id
                = ElevenLabs voice id). Audio is requested as pcm_16000 and
                resampled to 8 kHz. Any failure falls back to the local Piper
                default voice so the call never goes silent; a 401/403 disables
                further ElevenLabs attempts for the rest of the call.

The voices directory is configurable (constructor arg / env AI_AGENT_VOICES_DIR)
- there are no hardcoded install paths.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
Author: Tahir Almas. MIT licensed.
"""

import asyncio
import io
import logging
import os
import wave
from typing import AsyncIterator, Optional

import numpy as np

log = logging.getLogger("ai.tts")

# Voices dir: constructor arg wins, else env, else ./voices next to the sidecar.
DEFAULT_VOICES_DIR = os.environ.get(
    "AI_AGENT_VOICES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voices"),
)
DEFAULT_VOICE_NAME = "en_US-amy-medium"

TARGET_SAMPLE_RATE = 8000   # slin16 / AudioSocket
FRAME_BYTES        = 320    # 20 ms @ 8 kHz mono 16-bit

ELEVEN_URL_TMPL      = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
ELEVEN_DEFAULT_MODEL = "eleven_flash_v2_5"
ELEVEN_SAMPLE_RATE   = 16000  # we request output_format=pcm_16000

# PiperVoice.load takes 2.5-5.5 s (ONNX model load). Loading per call meant every
# answered call started with that much dead air before the greeting. Cache
# loaded voices process-wide, keyed by path.
_VOICE_CACHE: dict = {}
_VOICE_LOCK: Optional[asyncio.Lock] = None


def resolve_voice_path(voice_id: Optional[str], voices_dir: str,
                       default_voice: str = DEFAULT_VOICE_NAME) -> str:
    """Pick a voice .onnx file. Priority:
      1. {voices_dir}/{voice_id}.onnx     if voice_id given and the file exists
      2. {voices_dir}/{default_voice}.onnx if it exists
    Raises if neither is present."""
    voices_dir = os.path.abspath(voices_dir)
    for name in (voice_id, default_voice):
        if not name:
            continue
        candidate = os.path.join(voices_dir, f"{name}.onnx")
        if os.path.isfile(candidate):
            return candidate
        if name == voice_id:
            log.warning("voice_id=%r requested but %s not found", voice_id, candidate)
    raise RuntimeError(
        f"no Piper voice in {voices_dir} (tried {voice_id!r}, {default_voice!r}); "
        f"run download_voices.sh {default_voice}")


def _resample_linear(pcm16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resampler. Aliasing is acceptable for 8 kHz phone audio."""
    if src_sr == dst_sr:
        return pcm16
    ratio = dst_sr / src_sr
    out_len = int(len(pcm16) * ratio)
    if out_len <= 1:
        return np.zeros(0, dtype=np.int16)
    src_idx = np.linspace(0, len(pcm16) - 1, out_len, dtype=np.float32)
    lo = src_idx.astype(np.int32)
    hi = np.minimum(lo + 1, len(pcm16) - 1)
    frac = src_idx - lo
    out = (1.0 - frac) * pcm16[lo].astype(np.float32) + frac * pcm16[hi].astype(np.float32)
    return out.astype(np.int16)


def _iter_frames(pcm: np.ndarray):
    """Yield 8 kHz slin16 PCM as 20 ms (320-byte) frames, padding the tail."""
    view = pcm.tobytes()
    for off in range(0, len(view), FRAME_BYTES):
        chunk = view[off:off + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        yield chunk


class StreamingTTS:
    def __init__(self, provider: str, voice_id: str, model: str,
                 api_key: Optional[str] = None,
                 voices_dir: Optional[str] = None,
                 default_voice: str = DEFAULT_VOICE_NAME):
        self.provider      = (provider or "piper").strip().lower()
        self.voice_id      = voice_id
        self.model         = model
        self.api_key       = api_key or ""
        self.voices_dir    = voices_dir or DEFAULT_VOICES_DIR
        self.default_voice = default_voice or DEFAULT_VOICE_NAME
        self.voice_path    = None
        self._voice        = None
        self._native_sr    = None
        # ElevenLabs state (per call)
        self._eleven          = False   # provider active for this call
        self._eleven_disabled = False   # tripped on auth failure; piper thereafter
        self._client          = None    # httpx.AsyncClient, created in start()

    async def start(self) -> None:
        if self.provider == "elevenlabs":
            if self.api_key and self.voice_id:
                import httpx  # lazy: only needed for cloud TTS
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
                self._eleven = True
                log.info("TTS ready: elevenlabs voice=%s model=%s (piper fallback armed)",
                         self.voice_id, self.model or ELEVEN_DEFAULT_MODEL)
                return
            log.warning("tts provider=elevenlabs but %s missing; using piper",
                        "api_key" if not self.api_key else "voice_id")
        elif self.provider != "piper":
            log.warning("tts provider=%s not implemented; using piper", self.provider)
        await self._ensure_piper()
        log.info("TTS ready: voice=%s native_sr=%d",
                 os.path.basename(self.voice_path), self._native_sr)

    async def _ensure_piper(self) -> None:
        """Load (or fetch from cache) the local Piper voice."""
        global _VOICE_LOCK
        if self._voice is not None:
            return
        # In elevenlabs mode voice_id is a cloud id, not an .onnx basename, so
        # resolution lands on the configured/default local voice.
        piper_voice_id = None if self._eleven else self.voice_id
        self.voice_path = resolve_voice_path(piper_voice_id, self.voices_dir, self.default_voice)
        log.info("TTS voice file resolved: %s", self.voice_path)
        if _VOICE_LOCK is None:
            _VOICE_LOCK = asyncio.Lock()
        async with _VOICE_LOCK:
            voice = _VOICE_CACHE.get(self.voice_path)
            if voice is None:
                from piper import PiperVoice  # lazy import
                # Load is CPU-bound; offload so the event loop isn't blocked.
                voice = await asyncio.to_thread(PiperVoice.load, self.voice_path)
                _VOICE_CACHE[self.voice_path] = voice
                log.info("TTS voice loaded + cached: %s", os.path.basename(self.voice_path))
        self._voice = voice
        self._native_sr = self._voice.config.sample_rate

    async def _eleven_pcm(self, text: str) -> np.ndarray:
        """Synthesize via ElevenLabs; return PCM already resampled to 8 kHz."""
        url = ELEVEN_URL_TMPL.format(voice_id=self.voice_id)
        model = self.model if (self.model or "").startswith("eleven") else ELEVEN_DEFAULT_MODEL
        resp = await self._client.post(
            url,
            params={"output_format": f"pcm_{ELEVEN_SAMPLE_RATE}"},
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": model},
        )
        resp.raise_for_status()
        raw = resp.content
        pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
        return _resample_linear(pcm, ELEVEN_SAMPLE_RATE, TARGET_SAMPLE_RATE)

    async def _piper_pcm(self, text: str) -> np.ndarray:
        """Synthesize via local Piper; return PCM already resampled to 8 kHz."""
        await self._ensure_piper()

        def _synth() -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._voice.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf) as wf:
                return wf.readframes(wf.getnframes())

        # Serialize synthesis: piper phonemizes via espeak-ng, whose C API is
        # not thread-safe, and the PiperVoice may be shared across calls.
        async with _VOICE_LOCK:
            pcm_bytes = await asyncio.to_thread(_synth)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self._native_sr != TARGET_SAMPLE_RATE:
            pcm = _resample_linear(pcm, self._native_sr, TARGET_SAMPLE_RATE)
        return pcm

    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        """Yield slin16 PCM bytes in 20 ms (320-byte) chunks."""
        if not text.strip():
            return

        pcm = None
        if self._eleven and not self._eleven_disabled:
            try:
                pcm = await self._eleven_pcm(text)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403):
                    self._eleven_disabled = True
                    log.error("elevenlabs auth failed (%s); piper fallback for "
                              "the rest of this call", status)
                else:
                    log.warning("elevenlabs synthesis failed (%s); piper fallback "
                                "for this utterance", exc)

        if pcm is None:
            pcm = await self._piper_pcm(text)

        for chunk in _iter_frames(pcm):
            yield chunk

    async def close(self) -> None:
        self._voice = None   # cached copy stays in _VOICE_CACHE for reuse
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
