# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.1.3] - 2026-08-17

### Fixed
- The start of the first word of an utterance is no longer clipped. WebRTC VAD
  only calls a frame voiced once the talk-spurt carries enough energy, so a quiet
  onset (a leading fricative, the closure before a plosive) was discarded before
  the utterance buffer opened and the transcript began mid-word. The STT now keeps
  a rolling 300 ms window of pre-onset frames and prepends it when the utterance
  starts. It is most audible after barge-in, where the caller's first word
  competes with the agent still speaking. Suggested by `Asteriskdev` on
  r/Asterisk.

  One ring buffer per call, filled only while no utterance is open and drained at
  each talk-spurt, so the cost is fixed at 4.8 kB per call. The prepended audio is
  deliberately not counted towards `voice_ms`, which is what bounds the
  hallucination word-density check.

### Added
- `tests/test_lookback.py`, which drives `feed()` with a scripted VAD and asserts
  the pre-onset frames arrive at the transcriber, that the window stays bounded,
  and that `voice_ms` still counts only genuinely voiced audio.

## [0.1.2] - 2026-08-12

### Security
- The persona registry is now a strict allowlist. An AudioSocket connection whose
  UUID was never pre-registered by the dialplan is dropped instead of being served
  the `demo` persona, so the UUID acts as an authentication check against
  connections from unauthorized sources. Suggested by `ldo` on the Asterisk
  community forum.

### Fixed
- `TCP_NODELAY` is now set on every accepted AudioSocket connection. Outbound
  audio is one 320-byte frame every 20 ms, and Nagle's algorithm holds writes that
  small back waiting to coalesce them, so the pacing the writer works hard to get
  right could still be undone by the kernel. Reported by `crystalsighting` on
  r/Asterisk.

### Changed
- **Behaviour change:** if the dialplan's `POST /register` never reaches the
  sidecar (for example the `curl` times out), the call is now dropped rather than
  answered by the fallback persona.

### Added
- `tests/test_allowlist.py`, which drives `Call.run()` over a real UUID frame and
  checks both directions: unregistered UUIDs are rejected before the pipeline is
  built, and pre-registered ones still reach it with persona and caller intact.
  Wired into CI.
- `tests/test_nodelay.py`, which starts the real server on an ephemeral port,
  connects to it, and reads `TCP_NODELAY` back off the accepted socket. Wired
  into CI.
- README section on latency and network tuning.

## [0.1.0] - 2026-08-09

First public release. Extracted from the ICTContact AI Voice Agent, de-coupled
from the platform (multi-tenancy, billing, internal REST), and made config-driven.

### Added
- AudioSocket media sidecar: streaming STT → LLM → TTS over a single call.
- `agent.py` orchestrator: pre-register HTTP endpoint + AudioSocket server,
  reader/consumer tasks, sentence-buffered playback, barge-in.
- `stt.py`: OpenAI Whisper + ElevenLabs Scribe, WebRTC VAD end-of-utterance
  detection, DTX/silence-gap watchdog, hallucination filtering.
- `llm.py`: Anthropic Claude, streamed, with `tool_use`.
- `tts.py`: Piper (local) + ElevenLabs (cloud), resampled to 8 kHz slin16,
  process-wide voice cache; voices directory is configurable (no hardcoded paths).
- `tools.py`: default tool specs (transfer, schedule_callback, mark_dnc,
  crm_lookup, crm_update, send_sms), relayed to a webhook you control.
- Per-agent personas in YAML; Asterisk dialplan include; Docker + compose;
  Piper voice downloader.

### Known limitations
- v0.1: verified by construction, not yet exercised on a live call end-to-end.
- Tool *behaviour* is external, so you implement the webhook.
- TTS/STT/LLM providers beyond those listed require implementing the module
  interface (see PORTING.md).

[0.1.2]: https://github.com/ictinnovations/asterisk-ai-voice-agent/releases/tag/v0.1.2
[0.1.0]: https://github.com/ictinnovations/asterisk-ai-voice-agent/releases/tag/v0.1.0
