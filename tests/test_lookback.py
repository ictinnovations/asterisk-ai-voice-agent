"""Feeds a scripted talk-spurt and checks the pre-onset frames reach the STT.

The VAD only calls a frame voiced once the talk-spurt carries enough energy, so
without a lookback buffer the quiet head of the first word is dropped and the
transcript starts mid-word. The VAD is stubbed here so the onset lag is exact:
the assertion is about buffering, not about webrtcvad's tuning.

Run: python tests/test_lookback.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.stt import (
    FRAME_BYTES, LOOKBACK_FRAMES, LOOKBACK_MS, MIN_SILENCE_MS, StreamingSTT,
)


class _ScriptedVad:
    """is_speech() replays a fixed voiced/unvoiced script."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def is_speech(self, _frame, _rate):
        voiced = self.script[self.i] if self.i < len(self.script) else False
        self.i += 1
        return voiced


def _frame(n: int) -> bytes:
    """A frame whose every byte identifies its position in the stream."""
    return bytes([n % 256]) * FRAME_BYTES


async def main() -> None:
    closing = -(-MIN_SILENCE_MS // 20)   # silent frames needed to end an utterance
    quiet, voiced, trailing = 40, 30, closing + 5
    script = [False] * quiet + [True] * voiced + [False] * trailing
    assert quiet > LOOKBACK_FRAMES, "need more room tone than the lookback window"
    assert len(script) < 256, "frame ids must stay unique"

    stt = StreamingSTT(provider="whisper", model="whisper-1", language="en",
                       api_key="test")
    stt._vad = _ScriptedVad(script)

    seen = {}

    async def _capture(pcm, voice_ms):
        seen["pcm"], seen["voice_ms"] = pcm, voice_ms
        return ""

    stt._transcribe = _capture

    for i in range(len(script)):
        await stt.feed(_frame(i))

    assert "pcm" in seen, "utterance never closed"
    pcm = seen["pcm"]

    head = b"".join(_frame(i) for i in range(quiet - LOOKBACK_FRAMES, quiet))
    assert pcm.startswith(head), "pre-onset frames were not prepended"

    # Bounded window: older room tone must not be dragged in with it.
    assert len(pcm) == (LOOKBACK_FRAMES + voiced + closing) * FRAME_BYTES, \
        f"unexpected utterance length {len(pcm)} bytes"

    # The hallucination word budget must still count only genuinely voiced audio.
    assert seen["voice_ms"] == voiced * 20, seen["voice_ms"]

    await stt.close()
    print(f"ok: {LOOKBACK_FRAMES} pre-onset frames ({LOOKBACK_MS} ms) prepended, "
          f"voice_ms unchanged at {seen['voice_ms']}")


if __name__ == "__main__":
    asyncio.run(main())
