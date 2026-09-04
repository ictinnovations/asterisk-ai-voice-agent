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
from websockets.asyncio.server import serve

from .transport import AudioSocketTransport, Transport, WebSocketTransport
from .stt import StreamingSTT
from .llm import LLM
from .tts import StreamingTTS
from .tools import TOOL_SPECS

logging.basicConfig(
    level=os.environ.get("AI_AGENT_LOG", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("ai.agent")

# Ship TTS in natural chunks while the LLM is still streaming.
#
# The unit is the sentence. Splitting on every comma as well sounded choppy,
# because the TTS loses the run of the sentence at each cut, so clause breaks are
# used only once a sentence has run long enough that waiting for its end would
# cost more than the prosody. Terminators cover Latin, the Urdu full stop and
# question mark, and the Devanagari danda so non-English personas chunk too. A
# terminator counts only when whitespace follows it, so "3.5" and "e.g." do not
# fire mid-token, and a full stop after an abbreviation, an initial or a list
# number is not a sentence end either: "Dr. Smith", "J. Smith", "1. First".
_SENT_END = re.compile(r"[.!?؟।۔]+[\"'”’)\]]*\s")
_CLAUSE_END = re.compile(r"[,;:]\s(?=\S)")
_MD_STRIP = re.compile(r"[*_`#~]+")
LONG_SENTENCE_CHARS = 120
_ABBREVIATIONS = frozenset("""
mr mrs ms dr prof sr jr st mt ft rd ave blvd inc ltd co corp llc plc vs etc
e.g i.e cf al approx dept est fig gen gov hon rev sgt capt col lt cmdr
u.s u.k u.n a.m p.m no nos tel ext
jan feb mar apr jun jul aug sep sept oct nov dec
""".split())


def _abbreviation_before(text: str, dot: int) -> bool:
    """True if the full stop at text[dot] ends an abbreviation rather than a sentence."""
    m = re.search(r"(\S+)$", text[:dot])
    if not m:
        return False
    word = m.group(1).lstrip("(\"'“‘").lower()
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():          # an initial: "J. Smith"
        return True
    if word.isdigit():                             # a list number at line start: "1. First"
        before = text[:m.start()]
        return not before.strip() or before.endswith("\n")
    return False


def find_sentence_end(text: str) -> int:
    """Index just past the first sentence boundary in `text`, or -1 if none."""
    for m in _SENT_END.finditer(text):
        if m.group(0)[0] == "." and _abbreviation_before(text, m.start()):
            continue
        return m.end()
    return -1


def next_chunk(text: str) -> int:
    """Where to cut `text` for the next TTS chunk, or -1 to keep buffering.

    A sentence end wins. Failing that, once the buffer has run past
    LONG_SENTENCE_CHARS the last clause break is used, so a long sentence still
    starts playing before the model has finished it.
    """
    end = find_sentence_end(text)
    if end != -1:
        return end
    if len(text) >= LONG_SENTENCE_CHARS:
        last = -1
        for m in _CLAUSE_END.finditer(text):
            last = m.end()
        return last
    return -1


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


# How far synthesis may run ahead of playback, in 20 ms frames. These frames sit
# in our own queue, not handed to the transport, so they can still be dropped on
# barge-in; depth here costs nothing in interrupt latency. Two seconds is enough
# to ride out a synthesis stall without rendering a whole turn eagerly.
LOOKAHEAD_FRAMES = 100

# How long close() waits for synthesis to wind down on its own before cancelling
# it. Once stopping, the loop exits by itself as soon as any in-flight render
# returns, so this only ever fires on a hung provider request.
CLOSE_TIMEOUT_SEC = 20.0


class _TurnSpeaker:
    """Overlaps synthesis with playback for one assistant turn.

    Speaking a sentence at a time meant `play()` returned only once the caller
    had heard sentence N, and only then did sentence N+1 start synthesising. Every
    boundary therefore carried a silent gap equal to the next sentence's synthesis
    time, on every sentence rather than just the first.

    Here the transport plays from a single frame queue for the whole turn while a
    synthesis task keeps it fed, so the gap closes to time-to-first-byte of the
    next sentence, and to nothing at all once synthesis stays ahead.

    It also remembers which sentences reached the transport, so that after a
    barge-in the conversation history can record what the caller heard rather
    than what the model wrote.
    """

    def __init__(self, tts, should_stop, lookahead: int = LOOKAHEAD_FRAMES):
        self._tts = tts
        self._should_stop = should_stop
        self._sentences: asyncio.Queue = asyncio.Queue()
        # Unbounded on purpose. Lookahead is enforced by _put gating on qsize
        # instead of by maxsize, so the end-of-turn sentinel can never be
        # refused: a dropped sentinel would leave frames() waiting for ever.
        self._frames: asyncio.Queue = asyncio.Queue()
        self._lookahead = lookahead
        self._task = None
        self._closing = False
        self._texts = []          # every sentence added, in order
        self._played_upto = -1    # index of the last sentence a frame was played from

    def start(self) -> None:
        self._task = asyncio.create_task(self._synth_loop())

    def add(self, text: str) -> None:
        self._texts.append(text)
        self._sentences.put_nowait(len(self._texts) - 1)

    def finish(self) -> None:
        """No more sentences. Playback ends once the queue drains."""
        self._sentences.put_nowait(None)

    def spoken_text(self) -> str:
        """The sentences the caller heard, to sentence granularity.

        A sentence counts once its first frame has gone to the transport, so the
        one that was cut off by a barge-in is included in full. That is the
        right side to err on: the model should know what it was saying.
        """
        return " ".join(self._texts[: self._played_upto + 1]).strip()

    def _stopping(self) -> bool:
        return self._closing or self._should_stop()

    async def _put(self, item) -> bool:
        """Queue a frame, holding synthesis back to the lookahead depth.

        Gates on qsize rather than a bounded queue so the queue itself can never
        reject a put, and stays responsive to barge-in while held back.
        """
        while self._frames.qsize() >= self._lookahead:
            if self._stopping():
                return False
            await asyncio.sleep(0.01)
        if self._stopping():
            return False
        self._frames.put_nowait(item)
        return True

    async def _synth_loop(self) -> None:
        try:
            while True:
                idx = await self._sentences.get()
                if idx is None or self._stopping():
                    break
                async for frame in self._tts.synthesise(self._texts[idx], self._stopping):
                    if not await self._put((idx, frame)):
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("synthesis failed mid-turn")
        finally:
            # Unblock the player whichever way we left the loop. The queue is
            # unbounded, so this always lands.
            self._frames.put_nowait(None)

    async def frames(self):
        """The whole turn as one stream of 20 ms frames."""
        while True:
            item = await self._frames.get()
            if item is None:
                return
            idx, frame = item
            if idx > self._played_upto:
                self._played_upto = idx
            yield frame

    async def close(self) -> None:
        """Let synthesis wind down, cancelling only if it will not.

        Cancelling outright was wrong on the Piper path. The render runs in a
        worker thread under the process-wide voice lock, and a cancel lands on
        the await inside that lock's context, releasing it while espeak-ng is
        still running in the thread. The next call's render then enters Piper
        concurrently, which is the thread-safety problem the lock exists to
        prevent. So the loop is asked to stop and given time to notice: `_put`
        returns as soon as it polls, and `synthesise` discards a finished render
        unplayed. Cancel is kept only as a backstop for a provider that hangs.
        """
        if self._task is None:
            return
        self._closing = True
        if not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), CLOSE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                log.warning("synthesis did not stop within %.0f s; cancelling it",
                            CLOSE_TIMEOUT_SEC)
                self._task.cancel()
            except Exception:
                pass
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass


class Call:
    """One AI conversation over a single media connection."""

    def __init__(self, transport: Transport, cfg: dict, personas: dict, registry: Registry):
        self.transport = transport
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

    # ---- outbound audio ----------------------------------------------------
    def _should_stop(self) -> bool:
        """Barge-in or hangup. The transport polls this to cut playback short."""
        return self.interrupt or self.ended

    async def _play(self, pcm_iter):
        self.speaking = True
        try:
            await self.transport.play(pcm_iter, self._should_stop)
        except (ConnectionResetError, BrokenPipeError):
            self.ended = True
        finally:
            self.speaking = False

    async def say(self, text: str) -> None:
        text = tts_sanitize(text)
        if not text or self.tts is None or self.ended:
            return
        self._transcript.append({"role": "agent", "text": text})
        # Pass the stop check into synthesis too. The transport only polls it
        # between frames, which cannot happen until synthesis has produced one.
        await self._play(self.tts.synthesise(text, self._should_stop))

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
        # 1. transport handshake + persona lookup
        self.uuid = await self.transport.handshake()
        if not self.uuid:
            return
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
            chunk = await self.transport.read_audio()
            if chunk is None:
                return
            if not chunk:
                continue
            await self.stt.feed(chunk)
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
        tool calls, run them, then loop for the model's follow-up turn.

        Sentences go into a `_TurnSpeaker` rather than being played one at a time,
        so the next one is already synthesising while the current one plays.
        """
        buf = ""
        pending_tools = []
        speaker = _TurnSpeaker(self.tts, self._should_stop)
        speaker.start()
        player = asyncio.create_task(self._play(speaker.frames()))
        reply = self.llm.stream_reply()

        def queue(text: str) -> None:
            text = tts_sanitize(text)
            if not text or self.ended:
                return
            self._transcript.append({"role": "agent", "text": text})
            speaker.add(text)

        try:
            async for ev in reply:
                if self.interrupt or self.ended:
                    break
                if ev["kind"] == "text":
                    buf += ev["text"]
                    cut = next_chunk(buf)
                    while cut != -1:
                        sentence, buf = buf[:cut], buf[cut:]
                        queue(sentence)
                        cut = next_chunk(buf)
                elif ev["kind"] == "tool":
                    pending_tools.append(ev)
                elif ev["kind"] == "end":
                    if buf.strip():
                        queue(buf)
                        buf = ""
        finally:
            # Close the API stream now rather than whenever the generator is
            # collected, so an abandoned reply stops generating (and billing).
            await reply.aclose()
            # Let the queued audio finish before tool calls or the next turn.
            speaker.finish()
            try:
                await player
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("playback failed mid-turn")
            await speaker.close()
        if self.interrupt:
            # The caller cut in. History has to hold what they heard, not what
            # the model wrote. If the stream was abandoned there is no assistant
            # message at all, so the model would not know it had said anything;
            # if it completed, the message holds the whole reply plus any
            # tool_use blocks. Either way it becomes the spoken text alone. The
            # tool calls are dropped with it: a tool_result whose tool_use is no
            # longer in history is rejected by the API, and that rejection would
            # repeat on every later turn of the call.
            await self.llm.replace_last_assistant(speaker.spoken_text())
            if pending_tools:
                log.info("call %s: dropped %d tool call(s) from an interrupted turn",
                         self.uuid, len(pending_tools))
            return
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
        await self.transport.close()
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
                await Call(AudioSocketTransport(reader, writer), cfg, personas, registry).run()
            except Exception:
                log.exception("call handler crashed")

    # --- chan_websocket server (optional; Asterisk connects out to us) -----
    async def on_ws_call(ws):
        async with sem:
            try:
                await Call(WebSocketTransport(ws), cfg, personas, registry).run()
            except Exception:
                log.exception("websocket call handler crashed")

    reg_srv = await asyncio.start_server(http_register, host, int(listen.get("register_port", 9091)))
    as_srv = await asyncio.start_server(on_call, host, int(listen.get("audiosocket_port", 9092)))
    log.info("ai-voice-agent up: AudioSocket %s:%s, register %s:%s, personas=%s",
             host, listen.get("audiosocket_port", 9092),
             host, listen.get("register_port", 9091), list(personas))

    ws_port = listen.get("websocket_port")
    if not ws_port:
        async with reg_srv, as_srv:
            await asyncio.gather(reg_srv.serve_forever(), as_srv.serve_forever())
        return

    async with reg_srv, as_srv, await serve(on_ws_call, host, int(ws_port),
                                            subprotocols=["media"]) as ws_srv:
        log.info("chan_websocket listening on %s:%s", host, ws_port)
        await asyncio.gather(reg_srv.serve_forever(), as_srv.serve_forever(),
                             ws_srv.wait_closed())


def cli() -> None:
    """Console-script entry point (``asterisk-ai-voice-agent``)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
