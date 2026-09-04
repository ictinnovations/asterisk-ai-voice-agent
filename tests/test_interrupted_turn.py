"""Covers what a barge-in leaves behind in the conversation history.

Two silent failures. An interrupted turn used to leave no assistant message at
all, so the model never learned it had said anything. And if the model had
already asked for a tool before the caller cut in, the tool still ran and its
result was appended against a tool_use that was never recorded. The API rejects
that on the next request and on every request after it, so one barge-in at the
wrong moment left the caller hearing only the apology line for the rest of the
call.

Drives the real `LLM` and `Call` classes; only the Anthropic client, the TTS and
the transport are fakes.

Run: python tests/test_interrupted_turn.py
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent.agent import Call, Registry
from asterisk_ai_voice_agent.llm import LLM

FRAME = b"\x00" * 320
FRAMES_PER_SENTENCE = 10


# ---- fake Anthropic stream --------------------------------------------------

def _text(*parts):
    ev = [NS(type="content_block_start", content_block=NS(type="text"))]
    ev += [NS(type="content_block_delta", delta=NS(type="text_delta", text=p)) for p in parts]
    ev.append(NS(type="content_block_stop"))
    return ev


def _tool(tid, name, json_args):
    return [
        NS(type="content_block_start", content_block=NS(type="tool_use", id=tid, name=name)),
        NS(type="content_block_delta", delta=NS(type="input_json_delta", partial_json=json_args)),
        NS(type="content_block_stop"),
    ]


def _end(reason):
    return [NS(type="message_delta", delta=NS(stop_reason=reason))]


class _FakeStream:
    def __init__(self, events, delay):
        self.events, self.delay = list(events), delay
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        return self.events.pop(0)


class _FakeClient:
    """Each stream() call plays the next scripted reply."""

    def __init__(self, scripts, delay=0.0):
        self.scripts, self.delay = list(scripts), delay
        self.streams = []

    @property
    def messages(self):
        return self

    def stream(self, **kwargs):
        s = _FakeStream(self.scripts.pop(0), self.delay)
        self.streams.append(s)
        return s

    async def close(self):
        pass


# ---- fake TTS + transport ---------------------------------------------------

class _FakeTTS:
    async def synthesise(self, text, should_stop=None):
        await asyncio.sleep(0.02)
        for _ in range(FRAMES_PER_SENTENCE):
            if should_stop is not None and should_stop():
                return
            yield FRAME


class _FakeTransport:
    """Plays frames at a fixed pace; optionally flips the call's interrupt flag
    after N frames, which is what the reader task does on barge-in."""

    def __init__(self, interrupt_after=None):
        self.interrupt_after = interrupt_after
        self.call = None
        self.played = 0

    async def play(self, pcm, should_stop):
        async for _ in pcm:
            if should_stop():
                return
            self.played += 1
            if self.interrupt_after is not None and self.played == self.interrupt_after:
                self.call.interrupt = True
            await asyncio.sleep(0.005)

    async def close(self):
        pass


def _make(scripts, interrupt_after=None, delay=0.0):
    transport = _FakeTransport(interrupt_after)
    call = Call(transport, {}, {}, Registry())
    transport.call = call
    call.uuid = "test"
    call.tts = _FakeTTS()
    llm = LLM("anthropic", "m", 0.4, "", api_key="k")
    llm._client = _FakeClient(scripts, delay)
    call.llm = llm
    call.tool_calls = []

    async def run_tool(name, args):
        call.tool_calls.append((name, args))
        return {"ok": True}
    call.run_tool = run_tool
    return call, llm, transport


def _roles(history):
    return [m["role"] for m in history]


def _has_tool_result(history):
    return any(isinstance(m["content"], list)
               and any(b.get("type") == "tool_result" for b in m["content"])
               for m in history)


# ---- tests -----------------------------------------------------------------

def test_interrupt_mid_stream_after_a_tool_call():
    """The stream is abandoned after the model has asked for a tool."""
    script = (_text("Let me check ", "that for you. ")
              + _tool("toolu_1", "lookup", '{"q": "x"}')
              + _text(*(["and "] * 30))          # keeps the stream open
              + _end("tool_use"))
    call, llm, transport = _make([script], interrupt_after=3, delay=0.01)
    asyncio.run(call._llm_turn("hello"))

    assert call.tool_calls == [], f"tool ran after the interrupt: {call.tool_calls}"
    assert _roles(llm.history) == ["user", "assistant"], _roles(llm.history)
    assert llm.history[-1]["content"] == "Let me check that for you.", llm.history[-1]
    assert not _has_tool_result(llm.history)
    assert llm._client.streams[0].closed, "abandoned API stream was not closed"
    # The next caller turn must land as its own user message, not be merged
    # into a stale one.
    asyncio.run(llm.add_user("next"))
    assert _roles(llm.history) == ["user", "assistant", "user"], _roles(llm.history)
    print("ok: mid-stream interrupt kept the spoken text, dropped the tool, closed the stream")


def test_interrupt_during_playback_after_the_stream_completed():
    """The model finished, including a tool_use, but the caller cut in while
    sentence two was still playing."""
    script = (_text("One. ", "Two. ", "Three. ")
              + _tool("toolu_2", "lookup", "{}")
              + _end("tool_use"))
    call, llm, transport = _make([script], interrupt_after=FRAMES_PER_SENTENCE + 2)
    asyncio.run(call._llm_turn("hello"))

    assert call.tool_calls == []
    assert _roles(llm.history) == ["user", "assistant"], _roles(llm.history)
    assert llm.history[-1]["content"] == "One. Two.", llm.history[-1]
    assert not _has_tool_result(llm.history)
    print("ok: playback interrupt truncated history to the two sentences heard")


def test_uninterrupted_turn_runs_tools_and_keeps_full_history():
    first = _text("Checking. ") + _tool("toolu_3", "lookup", '{"q": "y"}') + _end("tool_use")
    second = _text("Done. ") + _end("end_turn")
    call, llm, transport = _make([first, second])
    asyncio.run(call._llm_turn("hello"))

    assert call.tool_calls == [("lookup", {"q": "y"})], call.tool_calls
    assert _roles(llm.history) == ["user", "assistant", "user", "assistant"], _roles(llm.history)
    blocks = llm.history[1]["content"]
    assert any(b["type"] == "tool_use" and b["id"] == "toolu_3" for b in blocks), blocks
    assert _has_tool_result(llm.history)
    assert transport.played == 2 * FRAMES_PER_SENTENCE, transport.played
    print("ok: an uninterrupted turn is unchanged: tool ran, history intact")


def test_interrupt_with_nothing_spoken_leaves_no_assistant_message():
    """Interrupted before the first sentence even finished streaming."""
    script = _text(*(["word "] * 40)) + _end("end_turn")
    call, llm, transport = _make([script], delay=0.01)

    async def main():
        turn = asyncio.create_task(call._llm_turn("hello"))
        await asyncio.sleep(0.05)
        call.interrupt = True
        await turn
    asyncio.run(main())

    assert _roles(llm.history) == ["user"], _roles(llm.history)
    assert transport.played == 0
    print("ok: nothing heard means nothing recorded, and no empty assistant message")


if __name__ == "__main__":
    test_interrupt_mid_stream_after_a_tool_call()
    test_interrupt_during_playback_after_the_stream_completed()
    test_uninterrupted_turn_runs_tools_and_keeps_full_history()
    test_interrupt_with_nothing_spoken_leaves_no_assistant_message()
    print("\nall interrupted turn tests passed")
