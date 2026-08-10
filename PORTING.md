# Architecture & extension notes

The sidecar ships complete and runnable: `agent.py` (orchestrator),
`audiosocket.py` (wire codec), `stt.py`, `llm.py`, `tts.py`, and `tools.py`.
This doc explains the seams so you can swap providers or add tools.

## Module contracts

`agent.py` drives the providers through these interfaces. Any implementation
that honours them drops straight in.

### `stt.StreamingSTT`
```python
StreamingSTT(provider, model, language, api_key=None,
             elevenlabs_api_key=None, min_silence_ms=None)
  async def start(self) -> None
  async def feed(self, pcm320: bytes) -> None      # one 320-byte slin16 frame in
  async def stream(self) -> AsyncIterator[dict]    # yields {'text': ...} finals
  def drain_pending(self) -> list[dict]            # non-blocking backlog drain
  property voice_active -> bool                    # sustained talk-spurt (barge-in)
  async def finalize_text(self) -> str             # flush utterance in progress at hangup
  async def close(self) -> None
```
Not a true streaming API. It VAD-gates with `webrtcvad`, buffers an utterance,
and transcribes on end-of-speech (silence or a wall-clock frame gap, so far-end
DTX doesn't strand the last utterance). Whisper hallucinations on near-silence
are filtered. Included providers: OpenAI Whisper and ElevenLabs Scribe.

### `llm.LLM`
```python
LLM(provider, model, temperature, system_prompt,
    tool_specs=None, tools_enabled=None, api_key=None)
  async def start(self) -> None
  async def add_user(self, text: str) -> None
  async def add_tool_result(self, tool_use_id: str, result_obj) -> None
  async def stream_reply(self) -> AsyncIterator[dict]   # {'kind':'text'|'tool'|'end', ...}
  async def close(self) -> None
```
Anthropic Claude, streamed, with `tool_use`. To use OpenAI or another provider,
reimplement this class keeping the `stream_reply()` event shape, and nothing else
changes.

### `tts.StreamingTTS`
```python
StreamingTTS(provider, voice_id, model, api_key=None,
             voices_dir=None, default_voice="en_US-amy-medium")
  async def start(self) -> None
  def synthesise(self, text: str) -> AsyncIterator[bytes]  # yields 320-byte slin16 frames
  async def close(self) -> None
```
Piper (local ONNX) + ElevenLabs (cloud), resampled to 8 kHz slin16. Piper voices
are cached process-wide (load is 2.5–5.5 s) and synthesis is serialized behind a
lock, because espeak-ng's phonemizer isn't thread-safe. The voices directory comes from
config (`providers.piper.voices_dir`) or `AI_AGENT_VOICES_DIR`; no hardcoded paths.

## Adding a tool

1. Add a spec to `TOOL_SPECS` in `tools.py` (name → description + parameters).
2. List the name in a persona's `tools_enabled`.
3. Handle it in your `tools.webhook_url` endpoint and return a JSON result.

When the model calls the tool, `agent.py` POSTs
`{"session", "tool", "args", "caller"}` to your webhook and feeds the JSON
response back to the model as the tool result. All tool *behaviour* is yours,
the sidecar only advertises the schema and relays the call.

## Piper voices

```bash
./download_voices.sh en_US-amy-medium        # into ./voices (or $AI_AGENT_VOICES_DIR)
```
Browse voices at https://huggingface.co/rhasspy/piper-voices.
