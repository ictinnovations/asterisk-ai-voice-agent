"""Drives WebSocketTransport against a fake chan_websocket peer.

Covers the three things that differ from AudioSocket and would be silent if
wrong: the call id arrives in the handshake URI rather than a frame, playback
runs ahead of real time instead of metering frames, and barge-in is a
FLUSH_MEDIA command rather than simply ceasing to write.

Run: python tests/test_websocket_transport.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.transport import LOOKAHEAD_SEC, WebSocketTransport

CALL_UUID = "11111111-2222-3333-4444-555555555555"


class _Request:
    def __init__(self, path):
        self.path = path


class _FakeWS:
    """Stands in for an Asterisk chan_websocket connection."""

    def __init__(self, path, inbound):
        self.request = _Request(path)
        self.sent_audio = bytearray()
        self.commands = []
        self.closed = False
        self._inbound = list(inbound)

    async def recv(self):
        if not self._inbound:
            await asyncio.sleep(3600)
        item = self._inbound.pop(0)
        return item

    async def send(self, msg):
        if isinstance(msg, str):
            self.commands.append(json.loads(msg)["command"])
        else:
            self.sent_audio.extend(msg)

    async def close(self):
        self.closed = True


def _media_start(fmt="slin", size=320, ptime=20):
    return json.dumps({"event": "MEDIA_START", "format": fmt,
                       "optimal_frame_size": size, "ptime": ptime})


async def _chunks(n, size=320):
    for _ in range(n):
        yield b"\x01" * size


async def test_handshake_reads_uuid_from_uri():
    ws = _FakeWS(f"/media?uuid={CALL_UUID}&persona=sales", [_media_start()])
    t = WebSocketTransport(ws)
    assert await t.handshake() == CALL_UUID
    assert t.frame_bytes == 320 and t.byte_rate == 16000.0, (t.frame_bytes, t.byte_rate)
    print("ok: uuid taken from the handshake URI, geometry from MEDIA_START")


async def test_missing_uuid_is_rejected():
    t = WebSocketTransport(_FakeWS("/media", [_media_start()]))
    assert await t.handshake() == "", "a call with no ?uuid= must be refused"
    print("ok: connection without ?uuid= rejected")


async def test_plaintext_mode_is_rejected_with_a_clear_error():
    ws = _FakeWS(f"/media?uuid={CALL_UUID}", ["MEDIA_START connection_id:conn1"])
    assert await WebSocketTransport(ws).handshake() == ""
    print("ok: plain-text control format refused rather than misparsed")


async def test_play_runs_ahead_but_returns_in_real_time():
    ws = _FakeWS(f"/media?uuid={CALL_UUID}", [_media_start()])
    t = WebSocketTransport(ws)
    await t.handshake()

    # 3 s of audio: long enough that a metered writer could not have sent it all
    # up front, and long enough to exceed the lookahead window.
    frames = 150
    t0 = time.monotonic()
    sent_by = {}

    async def watched():
        async for c in _chunks(frames):
            sent_by[len(ws.sent_audio) + len(c)] = time.monotonic() - t0
            yield c

    await t.play(watched(), lambda: False)
    elapsed = time.monotonic() - t0

    assert len(ws.sent_audio) == frames * 320, len(ws.sent_audio)
    # Everything was handed over well before it could have been heard...
    handover = max(sent_by.values())
    assert handover < 3.0 - LOOKAHEAD_SEC + 0.5, f"queued too slowly: {handover:.2f}s"
    # ...but play() still returned only once the caller had heard it.
    assert 2.8 < elapsed < 3.6, f"play() returned after {elapsed:.2f}s, expected ~3s"
    assert ws.commands == [], ws.commands
    print(f"ok: 3.0s queued within {handover:.2f}s, play() returned at {elapsed:.2f}s")


async def test_bargein_flushes_the_queue():
    ws = _FakeWS(f"/media?uuid={CALL_UUID}", [_media_start()])
    t = WebSocketTransport(ws)
    await t.handshake()

    stop = {"v": False}
    asyncio.get_running_loop().call_later(0.4, lambda: stop.__setitem__("v", True))

    t0 = time.monotonic()
    await t.play(_chunks(500), lambda: stop["v"])   # 10 s of audio
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"barge-in took {elapsed:.2f}s"
    assert ws.commands == ["FLUSH_MEDIA"], ws.commands
    print(f"ok: barge-in cut playback at {elapsed:.2f}s and sent FLUSH_MEDIA")


async def test_read_audio_separates_media_from_control():
    ws = _FakeWS(f"/media?uuid={CALL_UUID}",
                 [_media_start(), b"\x02" * 320,
                  json.dumps({"event": "MEDIA_XOFF"}), b"\x03" * 320])
    t = WebSocketTransport(ws)
    await t.handshake()
    assert await t.read_audio() == b"\x02" * 320
    assert await t.read_audio() == b""          # control frame, not audio
    assert await t.read_audio() == b"\x03" * 320
    print("ok: binary frames are audio, text frames are control")


async def main():
    await test_handshake_reads_uuid_from_uri()
    await test_missing_uuid_is_rejected()
    await test_plaintext_mode_is_rejected_with_a_clear_error()
    await test_play_runs_ahead_but_returns_in_real_time()
    await test_bargein_flushes_the_queue()
    await test_read_audio_separates_media_from_control()


if __name__ == "__main__":
    asyncio.run(main())
