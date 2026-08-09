"""
STT: OpenAI Whisper + ElevenLabs Scribe.

Receives 8 kHz slin16 frames from AudioSocket, runs WebRTC VAD to detect
end-of-utterance, then POSTs the buffered audio to the transcription API.
Yields a final transcript per utterance (via stream() / drain_pending()).

Providers (persona.stt_provider):
  whisper / openai  - OpenAI audio.transcriptions.create (whisper-1).
  elevenlabs        - ElevenLabs Scribe (/v1/speech-to-text, scribe_v1);
                      needs an ElevenLabs API key. Any failure falls back to
                      Whisper for that utterance; a 401/403 disables Scribe
                      for the rest of the call.

Design notes
- There is no true streaming STT here; we VAD-gate instead.
- VAD frames are 20 ms (320 bytes) - matches the AudioSocket frame size.
- An utterance "ends" after MIN_SILENCE_MS of silence following speech.
- Both APIs accept WAV; we wrap the buffered PCM as a single WAV.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
Author: Tahir Almas. MIT licensed.
"""

import asyncio
import io
import logging
import re
import wave
from typing import AsyncIterator, List, Optional

import webrtcvad
from openai import AsyncOpenAI

log = logging.getLogger("ai.stt")

ELEVEN_STT_URL       = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVEN_DEFAULT_MODEL = "scribe_v1"

SAMPLE_RATE     = 8000   # slin16 / AudioSocket
FRAME_BYTES     = 320    # 20 ms @ 8 kHz mono 16-bit
MIN_VOICE_MS    = 200    # need at least this much voiced audio to count as an utterance
MIN_SILENCE_MS  = 550    # silence after voice that triggers end-of-utterance
MAX_UTTERANCE_MS = 30000 # hard cap (Whisper API per-request limit ~25 MB)
# Barge-in signal: this many CONSECUTIVE voiced frames (240 ms) before we
# report the caller as actively speaking. The sustained-run requirement filters
# out clicks/coughs so the agent isn't cut off by noise.
BARGE_VOICE_RUN_FRAMES = 12
# Whisper hallucinates short pleasantries ("Thank you.", ". .") on near-silence.
# Real speech can't pack more than ~8 words into each second of VOICED audio;
# transcripts denser than that are junk and get dropped.
MAX_WORDS_PER_VOICED_SEC = 8.0
# Whisper's well-known silence/noise hallucinations: broadcast/end-credit
# boilerplate a real caller never says. Grammatically valid (so the density
# check misses them) but must be dropped or the agent "replies" to phantoms.
HALLUCINATION_PHRASES = (
    "thankyouforwatching",
    "thanksforwatching",
    "pleasesubscribe",
    "pleaselikeandsubscribe",
)
# Far-end DTX / silence suppression: some endpoints stop sending RTP entirely
# during silence, so Asterisk delivers NO AudioSocket frames and the
# frame-driven silence counter freezes - the utterance never closes and caller
# speech is only transcribed at hangup. Any wall-clock gap between frames
# longer than this is itself treated as silence. Nominal cadence is one frame
# per 20 ms; 120 ms = 5+ consecutive missing frames.
FRAME_GAP_MS = 120


class StreamingSTT:
    def __init__(self, provider: str, model: str, language: str,
                 api_key: Optional[str] = None,
                 elevenlabs_api_key: Optional[str] = None,
                 min_silence_ms: Optional[int] = None):
        self.provider = (provider or "whisper").strip().lower()
        self.model    = model
        self.language = language
        self.api_key  = api_key
        self.elevenlabs_api_key = elevenlabs_api_key or ""
        self.min_silence_ms = int(min_silence_ms) if min_silence_ms else MIN_SILENCE_MS
        self._vad     = None
        self._client  = None
        # ElevenLabs Scribe state (per call)
        self._eleven          = False   # provider active for this call
        self._eleven_disabled = False   # tripped on auth failure; whisper thereafter
        self._http            = None    # httpx.AsyncClient, created in start()
        # State
        self._buf       = bytearray()
        self._voice_ms  = 0
        self._silence_ms = 0
        self._voice_run = 0          # consecutive voiced frames (barge-in signal)
        self._utterance_started = False
        self._out_queue: asyncio.Queue = asyncio.Queue()
        # Wall-clock arrival time of the last frame (loop.time()); drives DTX
        # gap detection in feed() and _gap_watchdog().
        self._last_frame_t: Optional[float] = None
        self._gap_task: Optional[asyncio.Task] = None

    @property
    def voice_active(self) -> bool:
        """True while the caller is in a sustained talk-spurt (>=240 ms of
        consecutive voiced frames). Used by the orchestrator for barge-in."""
        return self._voice_run >= BARGE_VOICE_RUN_FRAMES

    def has_pending(self) -> bool:
        """True if finished transcripts are waiting in the output queue."""
        return self._out_queue.qsize() > 0

    def drain_pending(self) -> List[dict]:
        """Pop every transcript currently queued, without waiting. Lets the
        consumer coalesce a backlog into one LLM turn instead of replying to
        each stale utterance in sequence."""
        out: List[dict] = []
        while True:
            try:
                item = self._out_queue.get_nowait()
            except asyncio.QueueEmpty:
                return out
            if item is None:
                self._out_queue.put_nowait(None)  # close() sentinel - keep it
                return out
            out.append(item)

    async def start(self) -> None:
        if self.provider in ("elevenlabs", "scribe"):
            if self.elevenlabs_api_key:
                import httpx  # lazy: only needed for cloud Scribe
                self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
                self._eleven = True
            else:
                log.warning("stt provider=elevenlabs but api_key missing; using whisper")
        elif self.provider not in ("whisper", "openai"):
            log.warning("stt provider=%s not implemented; using openai/whisper", self.provider)
        if not self.api_key:
            if not self._eleven:
                raise RuntimeError("OpenAI API key not configured (providers.openai.api_key)")
            log.warning("no OpenAI key - Scribe active without whisper fallback")
        # WebRTC VAD: 0=least aggressive, 3=most. 2 is good for phone audio.
        self._vad = webrtcvad.Vad(2)
        if self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
        self._gap_task = asyncio.create_task(self._gap_watchdog())
        log.info("STT ready: provider=%s model=%s language=%s",
                 "elevenlabs" if self._eleven else "whisper", self.model, self.language)

    async def feed(self, samples: bytes) -> None:
        """Called for every 20 ms inbound audio frame. Drives VAD state."""
        now = asyncio.get_running_loop().time()
        if self._last_frame_t is not None and self._utterance_started:
            gap_ms = (now - self._last_frame_t) * 1000.0
            if gap_ms >= FRAME_GAP_MS:
                # Frames stopped flowing (DTX / sparse comfort noise): the
                # missing interval was silence the VAD never got to see.
                self._voice_run = 0
                self._silence_ms += int(gap_ms - 20)
                if self._voice_ms >= MIN_VOICE_MS and self._silence_ms >= self.min_silence_ms:
                    # Close the stalled utterance before this frame starts a new
                    # one, so post-gap speech isn't merged into it.
                    await self._flush()
        self._last_frame_t = now
        if len(samples) != FRAME_BYTES:
            if len(samples) < FRAME_BYTES:
                samples = samples + b"\x00" * (FRAME_BYTES - len(samples))
            else:
                samples = samples[:FRAME_BYTES]

        is_voice = False
        try:
            is_voice = self._vad.is_speech(samples, SAMPLE_RATE)
        except Exception as e:
            log.debug("vad error: %s", e)

        if is_voice:
            self._utterance_started = True
            self._voice_ms += 20
            self._voice_run += 1
            self._silence_ms = 0
            self._buf.extend(samples)
        elif self._utterance_started:
            self._voice_run = 0
            self._silence_ms += 20
            self._buf.extend(samples)   # keep trailing silence for a natural cutoff
            if self._voice_ms >= MIN_VOICE_MS and self._silence_ms >= self.min_silence_ms:
                await self._flush()

        # Hard cap so memory stays bounded if VAD never sees an end.
        if len(self._buf) > MAX_UTTERANCE_MS * SAMPLE_RATE * 2 // 1000:
            log.warning("MAX_UTTERANCE reached, flushing")
            await self._flush()

    def _take_buffer(self) -> tuple:
        """Grab the current utterance buffer and reset VAD state."""
        pcm = bytes(self._buf)
        voice_ms = self._voice_ms
        self._buf       = bytearray()
        self._voice_ms  = 0
        self._silence_ms = 0
        self._voice_run = 0
        self._utterance_started = False
        return pcm, voice_ms

    @staticmethod
    def _plausible(text: str, voice_ms: int) -> bool:
        """Reject Whisper hallucinations: punctuation-only output, or more words
        than a human can utter in the voiced audio we actually saw."""
        words = re.findall(r"[\w']+", text)
        if not words:
            return False
        squashed = re.sub(r"\s+", "", text).lower()
        for bad in HALLUCINATION_PHRASES:
            if bad in squashed:
                return False
        max_words = max(1.0, voice_ms / 1000.0 * MAX_WORDS_PER_VOICED_SEC)
        return len(words) <= max_words

    @staticmethod
    def _wrap_wav(pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        return buf.getvalue()

    async def _transcribe_eleven(self, wav_bytes: bytes) -> str:
        """POST WAV to ElevenLabs Scribe; return raw text (may raise)."""
        model = self.model if (self.model or "").startswith("scribe") else ELEVEN_DEFAULT_MODEL
        data = {"model_id": model}
        if self.language:
            data["language_code"] = self.language.split("-")[0]
        resp = await self._http.post(
            ELEVEN_STT_URL,
            headers={"xi-api-key": self.elevenlabs_api_key},
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

    async def _transcribe_whisper(self, wav_bytes: bytes) -> str:
        """POST WAV to OpenAI Whisper; return raw text or '' on error."""
        wav_buf = io.BytesIO(wav_bytes)
        wav_buf.name = "audio.wav"   # OpenAI SDK reads .name to set the MIME type
        try:
            resp = await self._client.audio.transcriptions.create(
                model=self.model if (self.model or "").startswith("whisper") else "whisper-1",
                file=wav_buf,
                language=self.language.split("-")[0] if self.language else None,
                response_format="json",
            )
            return (resp.text or "").strip()
        except Exception as e:
            log.error("Whisper API error: %s", e)
            return ""

    async def _transcribe(self, pcm: bytes, voice_ms: int) -> str:
        """Transcribe buffered PCM; return plausible text or ''."""
        wav_bytes = self._wrap_wav(pcm)
        log.debug("STT submit: %d bytes (%d ms voice)", len(pcm), voice_ms)

        text = None
        if self._eleven and not self._eleven_disabled:
            try:
                text = await self._transcribe_eleven(wav_bytes)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    self._eleven_disabled = True
                    log.error("elevenlabs STT auth failed (%s); whisper fallback "
                              "for the rest of this call", status)
                else:
                    log.warning("elevenlabs STT failed (%s); whisper fallback "
                                "for this utterance", e)

        if text is None:
            if self._client is None:
                return ""
            text = await self._transcribe_whisper(wav_bytes)

        if text and not self._plausible(text, voice_ms):
            log.info("STT dropped implausible transcript (%d ms voice): %r", voice_ms, text)
            return ""
        return text

    async def _flush(self) -> None:
        if not self._buf:
            return
        pcm, voice_ms = self._take_buffer()
        text = await self._transcribe(pcm, voice_ms)
        if text:
            await self._out_queue.put({"text": text, "is_final": True, "confidence": None})

    async def finalize_text(self) -> str:
        """Hangup-time flush: transcribe whatever utterance was in progress when
        the call ended (VAD never saw its trailing silence). Returns the text
        instead of queueing it - the consumer task is being torn down."""
        if not self._buf or self._voice_ms < MIN_VOICE_MS:
            self._take_buffer()
            return ""
        pcm, voice_ms = self._take_buffer()
        return await self._transcribe(pcm, voice_ms)

    async def _gap_watchdog(self):
        """Close utterances by wall clock when frames stop arriving.

        With full far-end DTX there may be NO further frames after the caller
        stops talking, so even the gap credit in feed() never runs (it needs a
        next frame to trigger). Poll: if an utterance is open and no frame has
        arrived for longer than the remaining silence budget, flush it. Also
        unsticks voice_active so barge-in doesn't read a vanished caller as
        still talking."""
        try:
            loop = asyncio.get_running_loop()
            while True:
                await asyncio.sleep(0.1)
                if not self._utterance_started or self._last_frame_t is None:
                    continue
                gap_ms = (loop.time() - self._last_frame_t) * 1000.0
                if gap_ms < FRAME_GAP_MS:
                    continue          # frames still flowing; feed() handles it
                self._voice_run = 0   # caller is not talking now
                if self._voice_ms >= MIN_VOICE_MS and self._silence_ms + gap_ms >= self.min_silence_ms:
                    log.info("frame gap %.0f ms - closing utterance "
                             "(far-end DTX/silence suppression)", gap_ms)
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def stream(self) -> AsyncIterator[dict]:
        """Yield final transcripts as they arrive. Terminates on close()."""
        while True:
            item = await self._out_queue.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._gap_task:
            self._gap_task.cancel()
        await self._out_queue.put(None)
        if self._client:
            await self._client.close()
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None
