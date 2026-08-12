"""
asterisk-ai-voice-agent - reference orchestrator (config-driven, no external
platform coupling).

Runs two servers:
  * an HTTP endpoint (register_port)   - the dialplan pre-registers persona by UUID
  * an AudioSocket TCP server (audiosocket_port) - Asterisk bridges the call here

Per AudioSocket connection:
  1. read the UUID frame -> look up the persona pre-registered for that UUID
  2. load the persona from personas.yaml
  3. speak the greeting (TTS -> paced AUDIO frames)
  4. two concurrent tasks for the life of the call:
       reader   : AUDIO frames -> stt.feed(); flag barge-in on stt.voice_active
       consumer : stt.stream() final transcripts -> LLM turn -> TTS -> paced out
  5. tool_use -> POST to tools.webhook_url -> feed result back to the LLM
  6. HANGUP / disconnect -> optional transcript POST -> cleanup

Provider modules: stt.py, llm.py, tts.py, tools.py alongside this file. The
AudioSocket framing is in audiosocket.py (mirrors the `asterisk-audiosocket`
npm package).

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
Author: Tahir Almas. MIT licensed.
"""

import asyncio
import json
import logging
import os
import re
import socket
import time
from pathlib import Path
from typing import Dict, Optional

import yaml
import httpx

from .audiosocket import Frame, FrameType, audio_frame, parse_uuid, FRAME_BYTES, FRAME_SEC, SILENCE_FRAME
from .stt import StreamingSTT
from .llm import LLM
from .tts import StreamingTTS
from .tools import TOOL_SPECS

logging.basicConfig(
    level=os.environ.get("AI_AGENT_LOG", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("ai.agent")

# Ship TTS in natural chunks while the LLM is still streaming. Includes Latin,
# Urdu full stop/question mark, and Devanagari danda so non-English personas
# chunk too.
_SENT_END = re.compile(r"([.!?؟।۔]\s|[,;:]\s(?=\S))")
_MD_STRIP = re.compile(r"[*_`#~]+")
PREROLL_FRAMES = 5   # silence frames before a talk-spurt so far-end codecs ramp up


def load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found (copy the matching .example.yaml)")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def tts_sanitize(text: str) -> str:
    """Strip markdown the TTS would mispronounce. Transcripts keep the raw text."""
    return _MD_STRIP.sub("", text).strip()


class Registry:
    """UUID -> pre-registered call context (persona name, caller, uniqueid)."""

    def __init__(self) -> None:
        self._by_uuid: Dict[str, dict] = {}

    def register(self, ctx: dict) -> None:
        uid = (ctx.get("uuid") or "").lower()
        if uid:
            self._by_uuid[uid] = ctx

    def take(self, uid: str) -> dict:
        return self._by_uuid.pop((uid or "").lower(), {})


class Call:
    """One AI conversation over a single AudioSocket connection."""

    def __init__(self, reader, writer, cfg: dict, personas: dict, registry: Registry):
        self.reader = reader
        self.writer = writer
        self.cfg = cfg
        self.personas = personas
        self.registry = registry
        self.uuid = ""
        self.persona: dict = {}
        self.stt: Optional[StreamingSTT] = None
        self.llm: Optional[LLM] = None
        self.tts: Optional[StreamingTTS] = None
        self.speaking = False
        self.interrupt = False
        self.ended = False
        self.started = time.monotonic()
        self._transcript = []

    # ---- outbound audio (paced) -------------------------------------------
    async def _paced_write(self, pcm_iter):
        """Write TTS frames metered to the 20 ms clock with a per-frame deadline
        clamp, so synthesis latency never causes a burst that overruns the far
        end's jitter buffer. This is the AudioSocket pacing gotcha, solved."""
        self.speaking = True
        try:
            for _ in range(PREROLL_FRAMES):
                self.writer.write(audio_frame(SILENCE_FRAME).encode())
            deadline = time.monotonic()
            async for chunk in pcm_iter:
                if self.interrupt or self.ended:   # barge-in / hangup: stop now
                    break
                if len(chunk) < FRAME_BYTES:
                    chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
                self.writer.write(audio_frame(chunk).encode())
                await self.writer.drain()
                deadline += FRAME_SEC
                now = time.monotonic()
                if deadline < now:                 # clamp: never schedule in the past
                    deadline = now
                await asyncio.sleep(deadline - now)
        except (ConnectionResetError, BrokenPipeError):
            self.ended = True
        finally:
            self.speaking = False

    async def say(self, text: str) -> None:
        text = tts_sanitize(text)
        if not text or self.tts is None or self.ended:
            return
        self._transcript.append({"role": "agent", "text": text})
        await self._paced_write(self.tts.synthesise(text))

    # ---- tool calling ------------------------------------------------------
    async def run_tool(self, name: str, args: dict) -> dict:
        url = (self.cfg.get("tools") or {}).get("webhook_url") or ""
        if not url:
            return {"ok": False, "error": "tools disabled (no webhook_url configured)"}
        headers = {"Content-Type": "application/json"}
        auth = (self.cfg.get("tools") or {}).get("webhook_auth")
        if auth:
            headers["Authorization"] = auth
        payload = {"session": self.uuid, "tool": name, "args": args,
                   "caller": self.persona.get("_caller")}
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            log.error("tool %s webhook failed: %s", name, e)
            return {"ok": False, "error": str(e)}

    # ---- main entry --------------------------------------------------------
    async def run(self) -> None:
        # 1. UUID frame + persona lookup
        try:
            first = await Frame.read(self.reader)
        except asyncio.IncompleteReadError:
            return
        if not first or first.type != FrameType.UUID:
            log.warning("expected UUID frame, got %s", first and first.type)
            return
        self.uuid = parse_uuid(first.payload)
        ctx = self.registry.take(self.uuid)
        if not ctx:
            # The dialplan pre-registers every UUID before it dials AudioSocket, so an
            # unknown one is a stale retry or an unauthorized connection, not a new call.
            log.warning("rejecting unregistered UUID %s", self.uuid)
            return
        pname = ctx.get("persona") or "demo"
        self.persona = dict(self.personas.get(pname) or self.personas.get("demo") or {})
        if not self.persona:
            log.error("no persona %r and no demo fallback; dropping call", pname)
            return
        self.persona["_caller"] = ctx.get("caller")
        log.info("call %s persona=%s caller=%s", self.uuid, pname, ctx.get("caller"))

        # 2. build the pipeline
        providers = self.cfg.get("providers", {})
        self.stt = StreamingSTT(
            provider=self.persona.get("stt_provider", "openai"),
            model=self.persona.get("stt_model", "whisper-1"),
            language=self.persona.get("stt_language", "en"),
            api_key=(providers.get("openai") or {}).get("api_key"),
            elevenlabs_api_key=(providers.get("elevenlabs") or {}).get("api_key"),
            min_silence_ms=self.persona.get("silence_timeout_ms"),
        )
        self.llm = LLM(
            provider=self.persona.get("llm_provider", "anthropic"),
            model=self.persona.get("llm_model", "claude-sonnet-4-6"),
            temperature=self.persona.get("llm_temperature", 0.4),
            system_prompt=self.persona.get("system_prompt", ""),
            tool_specs=TOOL_SPECS,
            tools_enabled=self.persona.get("tools_enabled", []),
            api_key=(providers.get("anthropic") or {}).get("api_key"),
        )
        piper = providers.get("piper") or {}
        self.tts = StreamingTTS(
            provider=self.persona.get("tts_provider", "piper"),
            voice_id=self.persona.get("tts_voice_id", piper.get("default_voice", "en_US-amy-medium")),
            model=self.persona.get("tts_model", ""),
            api_key=(providers.get("elevenlabs") or {}).get("api_key"),
            voices_dir=piper.get("voices_dir"),
            default_voice=piper.get("default_voice", "en_US-amy-medium"),
        )
        try:
            await asyncio.gather(self.stt.start(), self.llm.start(), self.tts.start())
        except Exception:
            log.exception("pipeline start failed for call %s", self.uuid)
            await self._teardown()
            return

        # 3. greeting
        greeting = self.persona.get("greeting")
        if greeting:
            await self.say(greeting)

        # 4. reader + consumer for the life of the call
        interrupt_enabled = bool(self.persona.get("interrupt_enabled", True))
        max_secs = int(self.persona.get("max_call_seconds", 900))
        consumer = asyncio.create_task(self._consumer_loop())
        try:
            await self._reader_loop(interrupt_enabled, max_secs)
        finally:
            self.ended = True
            self.interrupt = True
            await self.stt.close()      # pushes stream() sentinel -> consumer ends
            try:
                await asyncio.wait_for(consumer, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                consumer.cancel()
            await self._teardown()

    async def _reader_loop(self, interrupt_enabled: bool, max_secs: int) -> None:
        while not self.ended:
            if time.monotonic() - self.started > max_secs:
                log.info("call %s hit max_call_seconds", self.uuid)
                return
            try:
                frame = await Frame.read(self.reader)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            if frame is None or frame.type == FrameType.HANGUP:
                return
            if frame.type != FrameType.AUDIO:
                continue
            await self.stt.feed(frame.payload)
            if interrupt_enabled and self.speaking and self.stt.voice_active:
                self.interrupt = True   # caller talked over the agent

    async def _consumer_loop(self) -> None:
        async for item in self.stt.stream():
            text = (item or {}).get("text") or ""
            # Coalesce any backlog so we reply once, not to each stale utterance.
            for extra in self.stt.drain_pending():
                t = (extra or {}).get("text")
                if t:
                    text = f"{text} {t}".strip()
            if not text or self.ended:
                continue
            self.interrupt = False
            self._transcript.append({"role": "caller", "text": text})
            try:
                await self._llm_turn(text)
            except Exception:
                log.exception("llm turn failed for call %s", self.uuid)

    async def _llm_turn(self, user_text: str) -> None:
        await self.llm.add_user(user_text)
        await self._drive_llm()

    async def _drive_llm(self) -> None:
        """Stream one assistant turn: speak sentences as they form, collect any
        tool calls, run them, then loop for the model's follow-up turn."""
        buf = ""
        pending_tools = []
        async for ev in self.llm.stream_reply():
            if self.interrupt or self.ended:
                break
            if ev["kind"] == "text":
                buf += ev["text"]
                m = _SENT_END.search(buf)
                while m:
                    sentence, buf = buf[: m.end()], buf[m.end():]
                    await self.say(sentence)
                    m = _SENT_END.search(buf)
            elif ev["kind"] == "tool":
                pending_tools.append(ev)
            elif ev["kind"] == "end":
                if buf.strip():
                    await self.say(buf)
                    buf = ""
        for t in pending_tools:
            result = await self.run_tool(t["name"], t["args"])
            await self.llm.add_tool_result(t["id"], result)
        if pending_tools and not self.ended:
            await self._drive_llm()   # let the model continue after tool results

    async def _teardown(self) -> None:
        # Optional transcript POST for logging/analytics.
        url = (self.cfg.get("tools") or {}).get("webhook_url") or ""
        if url and self._transcript:
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    await c.post(url, json={"session": self.uuid, "event": "transcript",
                                            "turns": self._transcript})
            except Exception:
                pass
        for comp in (self.llm, self.tts):
            if comp:
                try:
                    await comp.close()
                except Exception:
                    pass
        try:
            self.writer.close()
        except Exception:
            pass
        log.info("call %s ended (%d turns)", self.uuid, len(self._transcript))


async def main() -> None:
    cfg = load_yaml(os.environ.get("AI_AGENT_CONFIG", "config.yaml"))
    personas = load_yaml(os.environ.get("AI_AGENT_PERSONAS", "personas.yaml"))
    listen = cfg.get("listen", {})
    host = listen.get("host", "127.0.0.1")
    registry = Registry()
    sem = asyncio.Semaphore(int((cfg.get("limits") or {}).get("max_concurrent_calls", 2)))

    # --- pre-register HTTP endpoint (dialplan curls persona here) ----------
    async def http_register(reader, writer):
        try:
            raw = await reader.read(65536)
            head, _, body = raw.partition(b"\r\n\r\n")
            if b"POST /register" in head:
                registry.register(json.loads(body or b"{}"))
                resp = b'{"ok":true}'
            else:
                resp = b'{"ok":false}'
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n%s" % (len(resp), resp))
            await writer.drain()
        except Exception as e:
            log.warning("register endpoint error: %s", e)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    # --- AudioSocket TCP server -------------------------------------------
    async def on_call(reader, writer):
        # Outbound audio is one 320-byte frame every 20 ms, which is exactly the
        # small-write pattern Nagle delays while it waits for more data to coalesce.
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as e:
                log.warning("could not set TCP_NODELAY: %s", e)
        async with sem:
            try:
                await Call(reader, writer, cfg, personas, registry).run()
            except Exception:
                log.exception("call handler crashed")

    reg_srv = await asyncio.start_server(http_register, host, int(listen.get("register_port", 9091)))
    as_srv = await asyncio.start_server(on_call, host, int(listen.get("audiosocket_port", 9092)))
    log.info("ai-voice-agent up: AudioSocket %s:%s, register %s:%s, personas=%s",
             host, listen.get("audiosocket_port", 9092),
             host, listen.get("register_port", 9091), list(personas))
    async with reg_srv, as_srv:
        await asyncio.gather(reg_srv.serve_forever(), as_srv.serve_forever())


def cli() -> None:
    """Console-script entry point (``asterisk-ai-voice-agent``)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
