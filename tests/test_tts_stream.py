"""Covers the two TTS fixes: streaming ElevenLabs (#1) and barge-in reaching
synthesis (#4).

Both were silent failures. The ElevenLabs one looked fine because audio did
arrive, just late. The barge-in one looked fine because playback did stop, just
after the whole sentence had already been rendered.

Run: python tests/test_tts_stream.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.tts import (
    FRAME_BYTES, TARGET_SAMPLE_RATE, StreamingTTS, _StreamResampler,
    _resample_linear,
)

ELEVEN_SR = 16000


def _fixed_step_reference(src, src_sr, dst_sr):
    """Resample at a constant step, which is what a stream must do.

    `_resample_linear` spreads its output across `linspace(0, N-1)`, so its step
    is (N-1)/(out-1) and depends on the total length. That is fine for a whole
    utterance but impossible for a stream, and over 16000 samples it drifts a
    full sample away from a fixed step. This is the correct oracle.
    """
    step = src_sr / dst_sr
    n = int(np.floor((len(src) - 1) / step)) + 1
    pos = step * np.arange(n)
    lo = pos.astype(np.int32)
    frac = (pos - lo).astype(np.float32)
    hi = np.minimum(lo + 1, len(src) - 1)
    return ((1.0 - frac) * src[lo].astype(np.float32)
            + frac * src[hi].astype(np.float32)).astype(np.int16)


def test_stream_resampler_matches_fixed_step_reference():
    """Fed in chunks, output must be identical to resampling in one go.

    A seam at each chunk boundary would still play, and only sound slightly
    wrong, so compare samples rather than trusting the ear. Random noise is used
    deliberately: on a smooth signal a one-sample drift is invisible.
    """
    rng = np.random.default_rng(7)
    src = (rng.standard_normal(16000) * 3000).astype(np.int16)
    want = _fixed_step_reference(src, ELEVEN_SR, TARGET_SAMPLE_RATE)

    for chunk in (160, 313, 1024, 4096):
        rs = _StreamResampler(ELEVEN_SR, TARGET_SAMPLE_RATE)
        got = [rs.feed(src[i:i + chunk]) for i in range(0, len(src), chunk)]
        got.append(rs.flush())
        got = np.concatenate([g for g in got if g.size])
        n = min(len(got), len(want))
        assert abs(len(got) - len(want)) <= 1, (
            f"chunk={chunk}: produced {len(got)} samples, expected {len(want)}")
        worst = int(np.max(np.abs(got[:n].astype(np.int32) - want[:n].astype(np.int32))))
        assert worst <= 1, f"chunk={chunk}: sample differs by {worst}"
    print("ok: chunked resampling is seam-free and matches a fixed-step resample")


def test_stream_resampler_preserves_pitch():
    """A 220 Hz tone in must still be 220 Hz out, fed in chunks.

    Checked against the signal itself rather than against `_resample_linear`.
    That function's step is (N-1)/(out-1), so it is very slightly off a true 2:1
    ratio and drifts a whole sample over 16000 samples. Inaudible, but it makes a
    poor oracle: agreeing with it would not mean the pitch was right.
    """
    t = np.arange(16000) / ELEVEN_SR
    src = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    rs = _StreamResampler(ELEVEN_SR, TARGET_SAMPLE_RATE)
    got = [rs.feed(src[i:i + 640]) for i in range(0, len(src), 640)]
    got.append(rs.flush())
    got = np.concatenate([g for g in got if g.size]).astype(np.float64)

    assert abs(len(got) - 8000) <= 2, f"produced {len(got)} samples, expected 8000"
    spec = np.abs(np.fft.rfft(got * np.hanning(len(got))))
    peak_hz = float(np.fft.rfftfreq(len(got), 1.0 / TARGET_SAMPLE_RATE)[int(np.argmax(spec))])
    assert abs(peak_hz - 220.0) < 2.0, f"peak at {peak_hz:.1f} Hz, expected 220"
    print(f"ok: 220 Hz tone resampled in chunks comes out at {peak_hz:.1f} Hz")


class _FakeStreamResponse:
    """Stands in for an httpx streaming response."""

    def __init__(self, chunks, status=200):
        self._chunks = chunks
        self.status_code = status
        self.aborted = True          # set False only if fully consumed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aread(self):
        return b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c
        self.aborted = False


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def stream(self, *a, **kw):
        return self._resp


def _tts_with(resp):
    t = StreamingTTS("elevenlabs", "voice", "eleven_flash_v2_5", api_key="k")
    t._eleven = True
    t._client = _FakeClient(resp)
    return t


def test_first_frame_arrives_before_the_whole_body():
    """The point of #1: a frame must be available from the first chunk alone."""
    # 0.5 s of 16 kHz audio, delivered as ten separate chunks.
    pcm = (np.sin(np.arange(8000) / 8.0) * 8000).astype(np.int16).tobytes()
    chunks = [pcm[i:i + 1600] for i in range(0, len(pcm), 1600)]
    resp = _FakeStreamResponse(chunks)
    tts = _tts_with(resp)

    async def main():
        agen = tts.synthesise("hello there")
        first = await agen.__anext__()
        assert len(first) == FRAME_BYTES, len(first)
        rest = [c async for c in agen]
        return 1 + len(rest)

    total = asyncio.run(main())
    # 8000 samples at 16 kHz -> 4000 at 8 kHz -> 8000 bytes -> 25 frames.
    assert total == 25, total
    print(f"ok: first frame yielded from the first chunk, {total} frames total")


def test_stop_during_stream_aborts_the_request():
    """Barge-in mid-stream must stop pulling, not just stop playing."""
    pcm = (np.zeros(80000) + 100).astype(np.int16).tobytes()
    chunks = [pcm[i:i + 1600] for i in range(0, len(pcm), 1600)]
    resp = _FakeStreamResponse(chunks)
    tts = _tts_with(resp)

    frames = []
    stop = {"v": False}

    async def main():
        async for c in tts.synthesise("a long sentence", lambda: stop["v"]):
            frames.append(c)
            if len(frames) >= 3:
                stop["v"] = True          # caller starts talking

    asyncio.run(main())
    assert len(frames) == 3, len(frames)
    assert resp.aborted, "stream was drained to the end instead of being abandoned"
    print("ok: stop during streaming abandoned the response after 3 frames")


def test_stop_before_synthesis_produces_nothing():
    resp = _FakeStreamResponse([b"\x00\x01" * 800])
    tts = _tts_with(resp)

    async def main():
        return [c async for c in tts.synthesise("hi", lambda: True)]

    assert asyncio.run(main()) == []
    assert resp.aborted, "the request was made despite an already-set stop"
    print("ok: stop set before synthesis makes no request at all")


def test_piper_render_is_discarded_when_interrupted():
    """Piper cannot be cancelled mid-render, but its output must not be played.

    The render runs in a worker thread under a process-wide lock, so abandoning
    the await would free the lock while espeak-ng still held the voice. The
    contract is therefore: finish the render, then throw it away.
    """
    tts = StreamingTTS("piper", "", "")
    tts._eleven = False
    rendered = {"n": 0}
    stop = {"v": False}

    async def fake_piper(text):
        stop["v"] = True                 # caller interrupts while we render
        await asyncio.sleep(0)
        rendered["n"] += 1
        return (np.zeros(4000) + 50).astype(np.int16)

    tts._piper_pcm = fake_piper

    async def main():
        return [c async for c in tts.synthesise("hello", lambda: stop["v"])]

    frames = asyncio.run(main())
    assert rendered["n"] == 1, "render should still have completed"
    assert frames == [], f"{len(frames)} frames played after an interrupt"
    print("ok: piper audio rendered during a barge-in is discarded, not played")


def test_no_stop_callable_still_works():
    """The parameter is optional; existing callers must be unaffected."""
    tts = StreamingTTS("piper", "", "")
    tts._eleven = False

    async def fake_piper(text):
        return (np.zeros(1600) + 7).astype(np.int16)

    tts._piper_pcm = fake_piper

    async def main():
        return [c async for c in tts.synthesise("hello")]

    frames = asyncio.run(main())
    assert len(frames) == 10, len(frames)
    print("ok: synthesise() without a stop callable behaves as before")


if __name__ == "__main__":
    test_stream_resampler_matches_fixed_step_reference()
    test_stream_resampler_preserves_pitch()
    test_first_frame_arrives_before_the_whole_body()
    test_stop_during_stream_aborts_the_request()
    test_stop_before_synthesis_produces_nothing()
    test_piper_render_is_discarded_when_interrupted()
    test_no_stop_callable_still_works()
    print("\nall tts stream tests passed")
