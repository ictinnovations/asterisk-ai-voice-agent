"""Drives Call.run() over a real UUID frame to prove the pre-registration allowlist holds.

Run: python tests/test_allowlist.py
"""

import asyncio
import struct
import sys
import uuid as _uuid
from pathlib import Path

# Running this file directly puts tests/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asterisk_ai_voice_agent import agent as ag
from asterisk_ai_voice_agent.audiosocket import FrameType
from asterisk_ai_voice_agent.transport import AudioSocketTransport

CALL_UUID = "11111111-2222-3333-4444-555555555555"


class _ReachedPipeline(Exception):
    """Raised by the patched STT constructor once the guard has been passed."""


class _Writer:
    def __init__(self):
        self.closed = False

    def write(self, _b):
        pass

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass

    def get_extra_info(self, *_a, **_k):
        return None


def _reader_with_uuid(value: str) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    payload = _uuid.UUID(value).bytes
    reader.feed_data(struct.pack(">BH", int(FrameType.UUID), len(payload)) + payload)
    reader.feed_eof()
    return reader


async def main() -> None:
    def _boom(**_kw):
        raise _ReachedPipeline()

    ag.StreamingSTT = _boom
    personas = {"demo": {"greeting": "hi"}, "sales": {"greeting": "hello"}}

    registry = ag.Registry()
    call = ag.Call(AudioSocketTransport(_reader_with_uuid(CALL_UUID), _Writer()),
                   {}, personas, registry)
    await call.run()
    assert call.persona == {}, f"unregistered UUID was served persona {call.persona!r}"
    print("ok: unregistered UUID rejected before the pipeline is built")

    registry = ag.Registry()
    registry.register({"uuid": CALL_UUID, "persona": "sales", "caller": "1001"})
    call = ag.Call(AudioSocketTransport(_reader_with_uuid(CALL_UUID), _Writer()),
                   {}, personas, registry)
    try:
        await call.run()
    except _ReachedPipeline:
        pass
    else:
        raise AssertionError("pre-registered UUID never reached the pipeline")
    assert call.persona.get("greeting") == "hello", call.persona
    assert call.persona.get("_caller") == "1001", call.persona
    print("ok: pre-registered UUID accepted, persona and caller carried through")


if __name__ == "__main__":
    asyncio.run(main())
