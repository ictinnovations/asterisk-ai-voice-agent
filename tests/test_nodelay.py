"""Starts the real server and checks the accepted AudioSocket socket has Nagle off.

Run: python tests/test_nodelay.py
"""

import asyncio
import os
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from asterisk_ai_voice_agent import agent as ag


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def main() -> None:
    as_port, reg_port = _free_port(), _free_port()
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "listen": {"host": "127.0.0.1", "audiosocket_port": as_port, "register_port": reg_port}
    }))
    personas = tmp / "personas.yaml"
    personas.write_text(yaml.safe_dump({"demo": {"greeting": "hi"}}))
    os.environ["AI_AGENT_CONFIG"] = str(cfg)
    os.environ["AI_AGENT_PERSONAS"] = str(personas)

    seen: asyncio.Queue = asyncio.Queue()

    class _Probe:
        def __init__(self, transport, *_a):
            sock = transport.writer.get_extra_info("socket")
            seen.put_nowait(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
            transport.writer.close()

        async def run(self):
            pass

    ag.Call = _Probe
    server = asyncio.create_task(ag.main())
    await asyncio.sleep(0.5)

    _, writer = await asyncio.open_connection("127.0.0.1", as_port)
    nodelay = await asyncio.wait_for(seen.get(), timeout=5)
    writer.close()
    server.cancel()

    assert nodelay != 0, f"TCP_NODELAY is off on the accepted socket (got {nodelay})"
    print("ok: accepted AudioSocket connection has TCP_NODELAY set")


if __name__ == "__main__":
    asyncio.run(main())
