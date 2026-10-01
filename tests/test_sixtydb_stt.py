"""Exercise 60db multipart uploads and the existing final-transcript pipeline.

Run: python tests/test_sixtydb_stt.py
"""

import asyncio
import io
import sys
import threading
import time
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from asterisk_ai_voice_agent import agent, stt as module
from asterisk_ai_voice_agent.stt import StreamingSTT

PCM = b"\x01\x00" * 8000


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        message = BytesParser(policy=default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n").encode() + body)
        parts = {part.get_param("name", header="content-disposition"): part
                 for part in message.iter_parts()}
        self.server.requests.append((self.headers.get("Authorization"), parts))
        time.sleep(self.server.delay)
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Location", "/redirected")
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        try:
            self.wfile.write(self.server.body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


async def configured(language="en-US", fallback=False):
    instance = StreamingSTT("sixtydb", "whisper-1", language,
                            api_key="fallback-test-key" if fallback else None,
                            sixtydb_api_key="local-test-key")
    await instance.start()
    return instance


async def run_checks(server):
    instance = await configured()
    fallback_calls = []
    async def fallback(wav_bytes):
        fallback_calls.append(wav_bytes)
        return "fallback text"
    try:
        assert instance._client is None, "OpenAI must be optional"
        instance._buf.extend(PCM)
        instance._voice_ms = 1000
        await instance._flush()
        assert instance.drain_pending() == [{"text": "hello caller", "is_final": True,
                                            "confidence": None}]
        authorization, parts = server.requests[-1]
        assert authorization == "Bearer local-test-key"
        assert parts["language"].get_payload(decode=True) == b"en"
        assert "model" not in parts and "model_id" not in parts
        assert parts["file"].get_filename() == "audio.wav"
        assert parts["file"].get_content_type() == "audio/wav"
        with wave.open(io.BytesIO(parts["file"].get_payload(decode=True))) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 8000)
            assert wav.readframes(wav.getnframes()) == PCM

        # Silence is a valid result; do not send it to another provider.
        instance._client = object()
        instance._transcribe_whisper = fallback
        server.body = b'{"text":"","warning_codes":["no_speech_detected"]}'
        assert await instance._transcribe(PCM, 1000) == ""
        assert fallback_calls == []
        # Errors and malformed responses use the configured fallback only.
        for body in (b'{"success":false,"text":"bad"}', b'{"error_code":"FAILED"}',
                     b'{"text":null}', b'{"text":12}', b'[]', b'not json'):
            server.body = body
            assert await instance._transcribe(PCM, 1000) == "fallback text"
        before = len(server.requests)
        server.status = 429
        assert await instance._transcribe(PCM, 1000) == "fallback text"
        assert len(server.requests) == before + 1, "no automatic retries"
        assert not instance._sixtydb_disabled
        server.status = 302
        assert await instance._transcribe(PCM, 1000) == "fallback text"
        assert len(server.requests) == before + 2, "no credential redirects"
        server.status = 401
        assert await instance._transcribe(PCM, 1000) == "fallback text"
        assert instance._sixtydb_disabled
        before = len(server.requests)
        assert await instance._transcribe(PCM, 1000) == "fallback text"
        assert len(server.requests) == before, "auth failure must stop further 60db requests"
        instance._client = None
        assert await instance._transcribe(PCM, 1000) == ""
    finally:
        instance._client = None
        http = instance._http
        await instance.close()
        assert http.is_closed and instance._http is None
        await asyncio.sleep(0)
        assert instance._gap_task.done()

    server.status = 200
    server.body = b'{"text":"hello caller"}'
    instance = await configured("auto")
    try:
        assert await instance._transcribe(PCM, 1000) == "hello caller"
        assert "language" not in server.requests[-1][1]
        # Existing hallucination guard still controls downstream output.
        server.body = b'{"text":"Thanks for watching"}'
        assert await instance._transcribe(PCM, 1000) == ""
        server.body = b'x' * (module.MAX_TRANSCRIPT_BYTES + 1)
        assert await instance._transcribe(PCM, 1000) == ""
    finally:
        await instance.close()

    for status in (403, 500):
        server.status = status
        instance = await configured()
        try:
            assert await instance._transcribe(PCM, 1000) == ""
            assert instance._sixtydb_disabled == (status == 403)
        finally:
            await instance.close()

    server.status = 200
    server.body = b'{"text":"hello caller"}'
    server.delay = 0.1
    instance = await configured()
    try:
        await instance._http.aclose()
        instance._http = httpx.AsyncClient(timeout=0.02)
        assert await instance._transcribe(PCM, 1000) == ""
        assert not instance._sixtydb_disabled
        instance._http.timeout = httpx.Timeout(1.0)
        before = len(server.requests)
        task = asyncio.create_task(instance._transcribe(PCM, 1000))
        deadline = asyncio.get_running_loop().time() + 2
        while len(server.requests) == before:
            assert asyncio.get_running_loop().time() < deadline, "request did not arrive"
            await asyncio.sleep(0.005)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation was swallowed")
    finally:
        await instance.close()
        server.delay = 0

    instance = StreamingSTT("sixtydb", "", "en")
    try:
        await instance.start()
    except RuntimeError as exc:
        assert "providers.sixtydb.api_key" in str(exc)
    else:
        raise AssertionError("missing key accepted")
    assert instance._http is None and instance._gap_task is None

    # Automatic detection must remain valid if the request falls back to Whisper.
    instance = StreamingSTT("sixtydb", "", "auto")
    whisper_args = {}
    async def whisper_create(**kwargs):
        whisper_args.update(kwargs)
        return SimpleNamespace(text="fallback text")
    instance._client = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=whisper_create)))
    assert await instance._transcribe_whisper(instance._wrap_wav(PCM)) == "fallback text"
    assert whisper_args["language"] is None
    instance._client = None
    await instance.close()

    # Constructor compatibility: min_silence_ms retains its old positional slot.
    legacy = StreamingSTT("openai", "whisper-1", "en", "key", None, 700)
    assert legacy.min_silence_ms == 700
    await legacy.close()

    # Drive actual Call.run() provider configuration, stopping before call IO.
    class ReachedSTT(Exception):
        pass
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        raise ReachedSTT()
    class Transport:
        async def handshake(self):
            return "local-call"
    original = agent.StreamingSTT
    agent.StreamingSTT = capture
    try:
        registry = agent.Registry()
        registry.register({"uuid": "local-call", "persona": "demo"})
        call = agent.Call(Transport(), {"providers": {"sixtydb": {"api_key": "configured-key"}}},
                          {"demo": {"stt_provider": "sixtydb"}}, registry)
        try:
            await call.run()
        except ReachedSTT:
            pass
        assert captured["provider"] == "sixtydb"
        assert captured["sixtydb_api_key"] == "configured-key"
        assert captured["api_key"] is None
    finally:
        agent.StreamingSTT = original
    print("ok: multipart WAV, queue, errors, auth, fallback, auto language, cleanup and call wiring")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.delay = 0
    server.status = 200
    server.body = b'{"text":"hello caller"}'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = module.SIXTYDB_STT_URL
    module.SIXTYDB_STT_URL = f"http://127.0.0.1:{server.server_port}/stt"
    try:
        asyncio.run(run_checks(server))
    finally:
        module.SIXTYDB_STT_URL = original
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
