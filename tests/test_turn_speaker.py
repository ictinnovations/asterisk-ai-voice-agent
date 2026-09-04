"""Covers _TurnSpeaker: synthesis must overlap playback across a whole turn (#2).

The failure this guards against is silent. Speaking a sentence at a time still
produced every word, just with a gap before each one, so the only way to catch a
regression is to time the frames rather than count them.

Run: python tests/test_turn_speaker.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.agent import LOOKAHEAD_FRAMES, _TurnSpeaker

FRAME = b"\x00" * 320
SYNTH_SECS = 0.30          # per sentence
# 500 ms of audio per sentence, deliberately longer than the 300 ms it costs to
# synthesise. That is the normal case: when synthesis is faster than playback the
# pipeline gets ahead and the boundary gap disappears. The reverse case is real
# too, and there the residual gap is the deficit, not a defect, so the assertion
# below is written against the deficit rather than against zero.
FRAMES_PER_SENTENCE = 25


class FakeTTS:
    """Synthesis that costs real time, so overlap is measurable."""

    def __init__(self, synth_secs=SYNTH_SECS, frames=FRAMES_PER_SENTENCE):
        self.synth_secs, self.frames = synth_secs, frames
        self.started = []
        self.rendered = 0

    async def synthesise(self, text, should_stop=None):
        self.started.append(time.monotonic())
        await asyncio.sleep(self.synth_secs)     # the render
        if should_stop is not None and should_stop():
            return
        self.rendered += 1
        for _ in range(self.frames):
            if should_stop is not None and should_stop():
                return
            yield FRAME


async def _drain(speaker, per_frame=0.02):
    """Stand in for a transport: pull frames at roughly real time."""
    out = []
    async for f in speaker.frames():
        out.append(time.monotonic())
        await asyncio.sleep(per_frame)
    return out


def test_next_sentence_synthesises_while_the_current_one_plays():
    tts = FakeTTS()

    async def main():
        sp = _TurnSpeaker(tts, lambda: False)
        sp.start()
        for i in range(3):
            sp.add(f"sentence {i}")
        sp.finish()
        t0 = time.monotonic()
        stamps = await _drain(sp)
        await sp.close()
        return t0, stamps

    t0, stamps = asyncio.run(main())
    assert len(stamps) == 3 * FRAMES_PER_SENTENCE, len(stamps)

    # Sentence 2's synthesis must begin before sentence 1 has finished playing.
    # Serialised, the second render could not start until 0.30 + 0.20 = 0.50 s.
    second = tts.started[1] - t0
    assert second < 0.45, f"second render started at {second:.2f}s, so it waited"

    # Compare against a serialised baseline computed on the same clock rather
    # than a wall-clock constant: asyncio.sleep granularity varies by platform
    # (about 31 ms for a 20 ms sleep on Windows) and would otherwise decide this.
    total = stamps[-1] - t0
    gaps = sorted(stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1))
    frame_interval = gaps[len(gaps) // 2]          # median, ignores the boundaries
    playback = len(stamps) * frame_interval
    serialised = 3 * SYNTH_SECS + playback         # render, play, render, play...
    overlapped = SYNTH_SECS + playback             # one render, then play through
    assert total < (serialised + overlapped) / 2, (
        f"turn took {total:.2f}s; serialised would be {serialised:.2f}s, "
        f"fully overlapped {overlapped:.2f}s")
    print(f"ok: second render began at {second:.2f}s; turn {total:.2f}s "
          f"vs {serialised:.2f}s serialised, {overlapped:.2f}s ideal")


def test_no_gap_between_sentences_once_synthesis_is_ahead():
    """The gap this closes is between the last frame of N and the first of N+1."""
    tts = FakeTTS()

    async def main():
        sp = _TurnSpeaker(tts, lambda: False)
        sp.start()
        for i in range(3):
            sp.add(f"sentence {i}")
        sp.finish()
        stamps = await _drain(sp)
        await sp.close()
        return stamps

    stamps = asyncio.run(main())
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    boundaries = [gaps[i * FRAMES_PER_SENTENCE - 1]
                  for i in range(1, len(stamps) // FRAMES_PER_SENTENCE)]
    worst = max(boundaries)

    # Measure the ordinary frame interval on this machine rather than assuming
    # 20 ms: asyncio.sleep granularity differs by platform (about 31 ms for a
    # 20 ms sleep on Windows, accurate on Linux) and would otherwise decide this.
    ordinary = sorted(gaps)[len(gaps) // 2]
    playback = FRAMES_PER_SENTENCE * ordinary
    deficit = max(0.0, SYNTH_SECS - playback)      # zero when synthesis keeps up
    budget = deficit + 2 * ordinary
    assert worst < budget, (
        f"boundary gap {worst * 1000:.0f} ms exceeds {budget * 1000:.0f} ms "
        f"(deficit {deficit * 1000:.0f} ms + 2 frames); synthesis is not ahead")
    print(f"ok: worst sentence-boundary gap {worst * 1000:.0f} ms against a "
          f"{SYNTH_SECS * 1000:.0f} ms render, budget {budget * 1000:.0f} ms")


def test_barge_in_stops_synthesis_and_playback():
    tts = FakeTTS()
    stop = {"v": False}

    async def main():
        sp = _TurnSpeaker(tts, lambda: stop["v"])
        sp.start()
        for i in range(6):
            sp.add(f"sentence {i}")
        sp.finish()
        got = []
        async for f in sp.frames():
            got.append(f)
            if len(got) == 12:            # partway through sentence 2
                stop["v"] = True
        await sp.close()
        return got

    got = asyncio.run(main())
    assert len(got) < 6 * FRAMES_PER_SENTENCE, "playback ran to the end of the turn"
    assert tts.rendered < 6, f"kept synthesising after the interrupt ({tts.rendered})"
    print(f"ok: barge-in stopped after {len(got)} frames, {tts.rendered}/6 sentences rendered")


def test_lookahead_is_bounded():
    """Synthesis must not render the whole turn eagerly into memory."""
    tts = FakeTTS(synth_secs=0.0, frames=80)

    async def main():
        sp = _TurnSpeaker(tts, lambda: False, lookahead=20)
        sp.start()
        for i in range(5):
            sp.add(f"sentence {i}")
        sp.finish()
        await asyncio.sleep(0.4)          # let synthesis run with nobody pulling
        depth = sp._frames.qsize()
        stop_after = []
        async for f in sp.frames():       # drain so close() is clean
            stop_after.append(f)
        await sp.close()
        return depth, len(stop_after)

    depth, total = asyncio.run(main())
    assert depth <= 21, f"queue grew to {depth}, lookahead was not honoured"
    assert total > 20, f"only {total} frames came through"
    print(f"ok: queue held at {depth} frames with a lookahead of 20")


def test_empty_turn_ends_cleanly():
    tts = FakeTTS()

    async def main():
        sp = _TurnSpeaker(tts, lambda: False)
        sp.start()
        sp.finish()
        got = [f async for f in sp.frames()]
        await sp.close()
        return got

    assert asyncio.run(main()) == []
    print("ok: a turn with no sentences ends without hanging")


if __name__ == "__main__":
    test_next_sentence_synthesises_while_the_current_one_plays()
    test_no_gap_between_sentences_once_synthesis_is_ahead()
    test_barge_in_stops_synthesis_and_playback()
    test_lookahead_is_bounded()
    test_empty_turn_ends_cleanly()
    print("\nall turn speaker tests passed")
