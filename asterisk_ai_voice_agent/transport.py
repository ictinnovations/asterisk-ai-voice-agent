"""
Media transports.

The orchestrator reaches Asterisk through one of these. The interface is shaped
around the one difference that actually matters between AudioSocket and
chan_websocket: who owns the playout clock.

Under AudioSocket we own it. We meter one 320-byte frame every 20 ms and must
never queue ahead, because audio handed to Asterisk is audio we cannot take back
when the caller interrupts. Queue depth therefore sets the barge-in floor.

Under chan_websocket Asterisk owns it, and FLUSH_MEDIA takes queued audio back.
Measured on 22.10.1: 18 s queued, flushed after 2 s, zero flushed bytes reached
the caller. So buffering ahead costs nothing there, and the two concerns come
apart.

Keeping `play()` responsible for pacing is what lets both live behind one
interface instead of leaking a 20 ms sleep into the orchestrator.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Callable, Optional
from urllib.parse import parse_qs, urlsplit

from websockets.exceptions import ConnectionClosed

from .audiosocket import (
    FRAME_BYTES,
    FRAME_SEC,
    SILENCE_FRAME,
    Frame,
    FrameType,
    audio_frame,
    parse_uuid,
)

log = logging.getLogger("ai.transport")

PREROLL_FRAMES = 5   # silence frames before a talk-spurt so far-end codecs ramp up

# How far ahead of the playout point WebSocketTransport is willing to queue.
# Asterisk stops accepting at 900 frames (18 s) and sends MEDIA_XOFF, so this
# sits an order of magnitude clear of the watermark. Two seconds is enough to
# ride out a TTS stall, and queueing deeper buys nothing: FLUSH_MEDIA discards
# the queue on barge-in whatever its depth.
LOOKAHEAD_SEC = 2.0


class Transport:
    """One media connection to Asterisk, for the life of a single call."""

    #: bytes of PCM per 20 ms frame; the TTS renders to this geometry
    frame_bytes: int = FRAME_BYTES

    async def handshake(self) -> str:
        """Return the call UUID, or "" to reject the connection."""
        raise NotImplementedError

    async def read_audio(self) -> Optional[bytes]:
        """Next inbound PCM chunk. b"" to skip, None when the call is over."""
        raise NotImplementedError

    async def play(self, pcm: AsyncIterator[bytes], should_stop: Callable[[], bool]) -> None:
        """Play a whole utterance, returning once the caller has heard it.

        Must poll `should_stop()` often enough to honour barge-in.
        """
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class AudioSocketTransport(Transport):
    """Asterisk 18+ over the AudioSocket TCP protocol.

    We are the pacer here, so `play()` blocks for the real duration of the
    utterance and barge-in costs at most one frame.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def handshake(self) -> str:
        try:
            first = await Frame.read(self.reader)
        except asyncio.IncompleteReadError:
            return ""
        if not first or first.type != FrameType.UUID:
            log.warning("expected UUID frame, got %s", first and first.type)
            return ""
        return parse_uuid(first.payload)

    async def read_audio(self) -> Optional[bytes]:
        try:
            frame = await Frame.read(self.reader)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None
        if frame is None or frame.type == FrameType.HANGUP:
            return None
        if frame.type != FrameType.AUDIO:
            return b""
        return frame.payload

    async def play(self, pcm: AsyncIterator[bytes], should_stop: Callable[[], bool]) -> None:
        for _ in range(PREROLL_FRAMES):
            self.writer.write(audio_frame(SILENCE_FRAME).encode())
        deadline = time.monotonic()
        async for chunk in pcm:
            if should_stop():
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

    async def close(self) -> None:
        try:
            self.writer.close()
        except Exception:
            pass


class WebSocketTransport(Transport):
    """Asterisk 20.18+/22.8+ over chan_websocket, where Asterisk dials *us*.

    Asterisk owns the playout clock, so `play()` hands audio over ahead of real
    time and then waits out the remainder, instead of metering frames itself.
    Barge-in is `FLUSH_MEDIA`, which drops whatever is still queued.

    Needs `control_message_format = json` in `chan_websocket.conf`; the driver
    defaults to a plain-text format we do not parse.
    """

    def __init__(self, ws):
        self.ws = ws
        self.byte_rate = 0.0
        self.fmt = ""

    async def handshake(self) -> str:
        # chan_websocket has no UUID frame. The dialplan supplies one as a query
        # parameter via the dial string's v() option:
        #   Dial(WebSocket/conn1/c(slin),v(uuid=${CALLUUID}))
        # It arrives with the HTTP handshake, so an unregistered call is refused
        # before any pipeline exists.
        query = parse_qs(urlsplit(self.ws.request.path).query)
        uuid = (query.get("uuid") or [""])[0]
        if not uuid:
            log.warning("websocket connection carried no ?uuid=; the dial string "
                        "needs the v(uuid=...) option")
            return ""
        try:
            started = await asyncio.wait_for(self._await_media_start(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("no MEDIA_START within 5s for %s", uuid)
            return ""
        return uuid if started else ""

    async def _await_media_start(self) -> bool:
        """Read up to MEDIA_START, which carries the frame geometry.

        Done here because nothing can be played before it lands, and because at
        this point in the call we are still the only reader of the socket.
        """
        while True:
            try:
                msg = await self.ws.recv()
            except ConnectionClosed:
                return False
            if isinstance(msg, bytes):
                continue                       # caller audio, before we can answer
            ev = self._control(msg)
            if ev is None:
                return False
            if ev.get("event") == "MEDIA_START":
                self.fmt = ev.get("format", "")
                self.frame_bytes = int(ev["optimal_frame_size"])
                self.byte_rate = self.frame_bytes * (1000.0 / int(ev["ptime"]))
                log.info("MEDIA_START format=%s frame=%dB ptime=%sms",
                         self.fmt, self.frame_bytes, ev["ptime"])
                return True

    def _control(self, msg: str) -> Optional[dict]:
        try:
            return json.loads(msg)
        except ValueError:
            log.error("chan_websocket is sending plain-text control messages; set "
                      "control_message_format = json in chan_websocket.conf")
            return None

    async def read_audio(self) -> Optional[bytes]:
        try:
            msg = await self.ws.recv()
        except ConnectionClosed:
            return None
        if isinstance(msg, bytes):
            return msg
        ev = self._control(msg)
        if ev is None:
            return None
        name = ev.get("event")
        if name == "MEDIA_XOFF":
            log.warning("MEDIA_XOFF: Asterisk stopped accepting audio")
        elif name == "ERROR":
            log.error("chan_websocket error: %s", ev.get("message") or ev)
        return b""

    async def play(self, pcm: AsyncIterator[bytes], should_stop: Callable[[], bool]) -> None:
        play_end = time.monotonic()   # when everything queued will have been heard
        async for chunk in pcm:
            if should_stop():
                return await self._flush()
            try:
                await self.ws.send(chunk)
            except ConnectionClosed:
                raise ConnectionResetError("websocket closed during playback")
            play_end += len(chunk) / self.byte_rate
            now = time.monotonic()
            if play_end < now:        # queue ran dry, so playout resumes from now
                play_end = now
            if not await self._wait(play_end - now - LOOKAHEAD_SEC, should_stop):
                return await self._flush()
        # Return only once the caller has actually heard it, so the orchestrator
        # knows when the agent stopped speaking.
        if not await self._wait(play_end - time.monotonic(), should_stop):
            await self._flush()

    async def _wait(self, seconds: float, should_stop: Callable[[], bool]) -> bool:
        """Sleep, giving up as soon as should_stop() goes true. False if cut short."""
        end = time.monotonic() + seconds
        while True:
            if should_stop():
                return False
            left = end - time.monotonic()
            if left <= 0:
                return True
            await asyncio.sleep(min(left, FRAME_SEC))

    async def _flush(self) -> None:
        try:
            await self.ws.send('{"command": "FLUSH_MEDIA"}')
        except ConnectionClosed:
            pass

    async def close(self) -> None:
        try:
            await self.ws.send('{"command": "HANGUP"}')
        except Exception:
            pass
        try:
            await self.ws.close()
        except Exception:
            pass
