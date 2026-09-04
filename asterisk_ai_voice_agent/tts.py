"""
TTS: Piper (local, ONNX) + ElevenLabs (cloud).

Synthesizes text to PCM, resamples to 8 kHz slin (the AudioSocket format),
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

TARGET_SAMPLE_RATE = 8000   # slin (16-bit, 8 kHz) / AudioSocket
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


class _StreamResampler:
    """Linear resampler that can be fed in chunks without a seam at each join.

    `_resample_linear` interpolates across a whole utterance. Calling it once per
    network chunk would restart the interpolation at every boundary and leave a
    small discontinuity there, so this keeps the unconsumed input samples and the
    fractional read position between calls.
    """

    def __init__(self, src_sr: int, dst_sr: int):
        self.step = (src_sr / dst_sr) if dst_sr else 1.0
        self.passthrough = (src_sr == dst_sr)
        self._tail = np.zeros(0, dtype=np.int16)
        self._t = 0.0                      # next output position, in input samples

    def feed(self, pcm16: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return pcm16
        buf = np.concatenate((self._tail, pcm16)) if self._tail.size else pcm16
        if buf.size < 2:
            self._tail = buf
            return np.zeros(0, dtype=np.int16)
        # Only positions whose upper neighbour is in the buffer can be produced,
        # so the last usable position is buf.size - 2, not buf.size - 1.
        n = int(np.floor((buf.size - 2 - self._t) / self.step)) + 1
        if n <= 0:
            self._tail = buf
            return np.zeros(0, dtype=np.int16)
        pos = self._t + self.step * np.arange(n, dtype=np.float64)
        lo = pos.astype(np.int32)
        frac = (pos - lo).astype(np.float32)
        out = ((1.0 - frac) * buf[lo].astype(np.float32)
               + frac * buf[lo + 1].astype(np.float32)).astype(np.int16)
        consumed = int(lo[-1])
        self._tail = buf[consumed:]
        self._t = self._t + self.step * n - consumed
        return out

    def flush(self) -> np.ndarray:
        """Emit any output position still inside the buffered input, then reset.

        Never extrapolates. At most one output sample (0.125 ms at 8 kHz) is
        dropped at the very end of a stream, which is better than inventing one:
        padding the tail with a copy of the last sample used to push a whole
        extra frame of near-silence onto the end of every utterance.
        """
        buf, self._tail = self._tail, np.zeros(0, dtype=np.int16)
        t, self._t = self._t, 0.0
        if self.passthrough or buf.size < 2:
            return np.zeros(0, dtype=np.int16)
        n = int(np.floor((buf.size - 1 - t) / self.step)) + 1
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        pos = t + self.step * np.arange(n, dtype=np.float64)
        lo = np.minimum(pos.astype(np.int32), buf.size - 1)
        hi = np.minimum(lo + 1, buf.size - 1)
        frac = (pos - lo).astype(np.float32)
        return ((1.0 - frac) * buf[lo].astype(np.float32)
                + frac * buf[hi].astype(np.float32)).astype(np.int16)


def _iter_frames(pcm: np.ndarray):
    """Yield 8 kHz slin PCM as 20 ms (320-byte) frames, padding the tail."""
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

    async def _eleven_frames(self, text: str, should_stop=None) -> AsyncIterator[bytes]:
        """Yield 8 kHz 320-byte frames as the ElevenLabs response arrives.

        The endpoint streams, so reading it incrementally puts first audio at
        roughly time-to-first-byte instead of after the whole utterance has been
        synthesised. Stopping early closes the HTTP stream, which is also how a
        barge-in stops us paying for audio nobody will hear.
        """
        url = ELEVEN_URL_TMPL.format(voice_id=self.voice_id)
        model = self.model if (self.model or "").startswith("eleven") else ELEVEN_DEFAULT_MODEL
        rs = _StreamResampler(ELEVEN_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        pending = b""
        odd = b""            # a chunk can split an int16 down the middle
        async with self._client.stream(
            "POST", url,
            params={"output_format": f"pcm_{ELEVEN_SAMPLE_RATE}"},
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": model},
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()          # body is needed for a useful message
                resp.raise_for_status()
            async for raw in resp.aiter_bytes():
                if should_stop is not None and should_stop():
                    return
                if not raw:
                    continue
                raw = odd + raw
                usable = len(raw) // 2 * 2
                odd = raw[usable:]
                if not usable:
                    continue
                out = rs.feed(np.frombuffer(raw[:usable], dtype=np.int16))
                if out.size:
                    pending += out.tobytes()
                    while len(pending) >= FRAME_BYTES:
                        yield pending[:FRAME_BYTES]
                        pending = pending[FRAME_BYTES:]
                        if should_stop is not None and should_stop():
                            return
        tail = rs.flush()
        if tail.size:
            pending += tail.tobytes()
        while len(pending) >= FRAME_BYTES:
            yield pending[:FRAME_BYTES]
            pending = pending[FRAME_BYTES:]
        if pending:
            yield pending + b"\x00" * (FRAME_BYTES - len(pending))

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

    async def synthesise(self, text: str, should_stop=None) -> AsyncIterator[bytes]:
        """Yield slin PCM bytes in 20 ms (320-byte) chunks.

        `should_stop` is polled around synthesis as well as between frames. The
        transports already poll it per frame, but that check cannot run until the
        first frame exists, so without this a barge-in during synthesis had no
        effect at all and the whole sentence was rendered anyway (issue #4).

        On the ElevenLabs path stopping aborts the HTTP stream. On the Piper path
        an in-flight render is *not* cancelled: it runs in a worker thread under
        a process-wide lock, and abandoning the await would release that lock
        while espeak-ng was still using the voice, which is the thread-safety
        problem the lock exists to prevent. Piper is checked either side of the
        render instead, so the audio is discarded rather than played.
        """
        if not text.strip():
            return
        if should_stop is not None and should_stop():
            return

        if self._eleven and not self._eleven_disabled:
            produced = False
            try:
                async for chunk in self._eleven_frames(text, should_stop):
                    produced = True
                    yield chunk
                return
            except Exception as exc:
                if produced:
                    # Some audio already reached the caller; restarting the
                    # sentence on piper would repeat what they just heard.
                    log.warning("elevenlabs stream failed mid-utterance (%s); "
                                "no fallback for this one", exc)
                    return
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403):
                    self._eleven_disabled = True
                    log.error("elevenlabs auth failed (%s); piper fallback for "
                              "the rest of this call", status)
                else:
                    log.warning("elevenlabs synthesis failed (%s); piper fallback "
                                "for this utterance", exc)

        if should_stop is not None and should_stop():
            return
        pcm = await self._piper_pcm(text)
        if should_stop is not None and should_stop():
            return          # interrupted while rendering; drop it unplayed

        for chunk in _iter_frames(pcm):
            yield chunk
            if should_stop is not None and should_stop():
                return

    async def close(self) -> None:
        self._voice = None   # cached copy stays in _VOICE_CACHE for reuse
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
