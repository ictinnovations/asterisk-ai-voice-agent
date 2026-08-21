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
import logging
import time
from typing import AsyncIterator, Callable, Optional

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
