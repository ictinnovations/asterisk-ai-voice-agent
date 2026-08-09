"""
Asterisk AudioSocket frame codec (Python).

Wire format (binary, big-endian, over TCP):
  Byte 0:    type   (1 byte)
  Bytes 1-2: length (2 bytes, big-endian, payload length only)
  Bytes 3+:  payload

Types:
  0x00 HANGUP  no payload
  0x01 UUID    16 bytes (call id from AudioSocket(<uuid>,host:port))
  0x10 AUDIO   slin16 8 kHz mono - 320 bytes per 20 ms frame
  0xff ERROR   1 byte error code

Reference: app_audiosocket.c in the Asterisk source tree.
Companion TypeScript implementation: `asterisk-audiosocket` on npm.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
"""

import asyncio
import struct
import uuid as _uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

FRAME_BYTES = 320   # 20 ms @ 8 kHz mono 16-bit
FRAME_SEC = 0.02
SAMPLE_RATE = 8000
SILENCE_FRAME = b"\x00" * FRAME_BYTES


class FrameType(IntEnum):
    HANGUP = 0x00
    UUID = 0x01
    AUDIO = 0x10
    ERROR = 0xff


@dataclass
class Frame:
    type: FrameType
    payload: bytes

    @classmethod
    async def read(cls, reader: asyncio.StreamReader) -> "Optional[Frame]":
        header = await reader.readexactly(3)
        ftype, length = struct.unpack(">BH", header)
        payload = await reader.readexactly(length) if length else b""
        try:
            ftype = FrameType(ftype)
        except ValueError:
            ftype = FrameType.ERROR
        return cls(type=ftype, payload=payload)

    def encode(self) -> bytes:
        return struct.pack(">BH", int(self.type), len(self.payload)) + self.payload


def audio_frame(samples: bytes) -> Frame:
    return Frame(type=FrameType.AUDIO, payload=samples)


def hangup_frame() -> Frame:
    return Frame(type=FrameType.HANGUP, payload=b"")


def parse_uuid(payload: bytes) -> str:
    if len(payload) != 16:
        return ""
    return str(_uuid.UUID(bytes=payload))
