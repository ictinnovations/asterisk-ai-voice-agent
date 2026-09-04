# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.5] - 2026-09-05

### Fixed
- Sentence N+1 now synthesises while sentence N is still playing. `say()` played
  one sentence at a time and returned only once the caller had heard it, so every
  sentence boundary carried a silent gap the length of the next sentence's
  synthesis, on every sentence rather than just the first. A turn now feeds one
  frame queue that the transport plays from continuously, with synthesis running
  ahead of playback up to a bounded lookahead. (#2)

  Measured on Asterisk 22.10.1 with `Echo()`, three sentences, using the arrival
  times of the echoed audio as the caller's experience of the turn:

  | per sentence | before | after |
  |---|---|---|
  | 0.5 s synthesis, 0.4 s audio | 2.65 s turn, two 500 ms holes | 1.88 s turn, 120 ms worst gap |
  | 0.25 s synthesis, 0.5 s audio | n/a | 1.74 s turn, 21 ms worst gap, no holes |

  Overlap cannot create throughput. Where a sentence costs more to synthesise
  than it takes to play, the pipeline falls behind by that difference on every
  sentence and the residual gap is the deficit; what goes away is the old gap,
  which was the entire synthesis time. Where synthesis is faster than playback,
  which is the usual case, the gap closes completely.

  The lookahead queue is unbounded with synthesis gated on its depth, rather than
  a bounded queue: a bounded one can refuse the end-of-turn sentinel when it is
  full, and a dropped sentinel leaves playback waiting for ever.

  A second effect worth knowing about: `speaking` now stays true across the whole
  turn instead of flickering false at each sentence boundary, so a caller who
  starts talking in a gap is detected as barge-in rather than missed.

### Added
- `tests/test_turn_speaker.py`, timing the frames rather than counting them,
  because a gap between sentences still delivers every word. Covers overlap,
  boundary gap, barge-in stopping both synthesis and playback, the lookahead
  bound, and an empty turn. Wired into CI.


### Notes
- Investigated and closed #3 (the process-wide Piper lock) without a code change.
  Moving synthesis into worker processes was measured against the lock in the
  shipped image with a real voice, four concurrent calls, five trials: 2.30 s
  against 1.99 s total, and the first caller waited 1.89 s instead of 0.48 s.
  Worse on every axis. ONNX Runtime already parallelises a single render across
  every core, so the lock serialises work that is already saturating the CPU, and
  the pool only adds contention, IPC, and the loss of FIFO ordering. Capping each
  worker to one ONNX thread was worse again, 5.41 s against 2.00 s. The ceiling is
  CPU, not the lock. Full numbers on the issue.

## [0.1.4] - 2026-09-04

### Fixed
- Barge-in now reaches synthesis. `Call._should_stop()` was polled only by the
  transports, inside the loop that consumes frames, so it could not run until
  synthesis had produced a frame. `StreamingTTS.synthesise()` renders a whole
  sentence before yielding its first one, which meant a caller who interrupted
  while we were still rendering had no effect at all: the sentence was built in
  full, then playback stopped at frame zero. The stop check is now passed into
  `synthesise()` and polled around synthesis as well as between frames.

  On the ElevenLabs path this closes the HTTP stream, so an interrupted sentence
  stops being billed. On the Piper path the render is deliberately *not*
  cancelled: it runs in a worker thread under the process-wide `_VOICE_LOCK`, and
  abandoning the await would release that lock while espeak-ng still held the
  voice, which is the thread-safety problem the lock exists to prevent. Piper is
  checked either side of the render instead, so the audio is discarded rather
  than played. (#4)
- ElevenLabs audio now streams. The `/stream` endpoint was already being called,
  but the response was consumed with `resp.content`, which waits for the whole
  body, so the caller heard nothing until the entire utterance had been
  synthesised. The response is now read with `aiter_bytes()`, resampled
  incrementally and yielded frame by frame, so first audio lands at roughly
  time-to-first-byte. (#1)

### Changed
- Outbound media now goes through a `Transport` interface (`transport.py`) instead
  of `Call` holding an asyncio reader/writer pair directly. `AudioSocketTransport`
  is the only implementation and behaviour is unchanged; the 20 ms metered writer
  moved from `Call._paced_write` to `AudioSocketTransport.play` verbatim.

  The split is at `play()` rather than at frame encoding, because that is where
  AudioSocket and `chan_websocket` genuinely differ: under AudioSocket we own the
  playout clock and must not queue ahead, since queued audio cannot be taken back
  on barge-in, whereas `chan_websocket` lets Asterisk pace and `FLUSH_MEDIA` takes
  queued audio back. Measured on Asterisk 22.10.1: 18 s queued, flushed after 2 s,
  and not one flushed byte reached the caller.

### Added
- `WebSocketTransport`, so the sidecar can also run over `chan_websocket` on
  Asterisk 20.18+/22.8+. Off unless `listen.websocket_port` is set; AudioSocket
  remains the default and both can run at once.

  Asterisk owns the playout clock here, so `play()` hands audio over ahead of
  real time and waits out the remainder rather than metering frames, and
  barge-in sends `FLUSH_MEDIA` instead of simply ceasing to write. It buffers
  2 s ahead: enough to ride out a TTS stall, and far clear of the 900-frame
  (18 s) `MEDIA_XOFF` watermark. Queueing deeper would cost nothing in barge-in
  latency but buys nothing either.

  `chan_websocket` has no UUID frame, and the `connection_id` is identical on
  every call, so the persona allowlist would have had nothing to key on. The
  call id comes from the dial string instead —
  `Dial(WebSocket/conn1/c(slin),v(uuid=${CALLUUID}))` — which arrives as a query
  parameter on the HTTP handshake, before any media. So an unregistered call is
  still refused before a pipeline is built, marginally earlier than AudioSocket
  manages. Verified on 22.10.1, along with the `Set(__FOO=)`/`Set(_FOO=)`
  distinction noted in the README.
- `tests/test_websocket_transport.py`, which drives the transport against a fake
  chan_websocket peer and asserts the three behaviours that would otherwise fail
  silently: the id is read from the handshake URI, playback runs ahead of real
  time yet `play()` still returns only once the caller has heard it, and
  barge-in emits `FLUSH_MEDIA`. Wired into CI.
- `packaging/systemd/asterisk-ai-voice-agent.service`, so a pip install can run
  as a service without Docker. Runs as a dedicated unprivileged system user,
  restarts on failure, and keeps `/etc/asterisk-ai-voice-agent` read-only to the
  process via `ProtectSystem=strict`, because the config file holds API keys.
  Piper voices live in a `StateDirectory=`, which is the only writable path.
  Requested by `crystalsighting` on r/Asterisk.

- `_StreamResampler`, a linear resampler that can be fed in chunks without a
  seam at each join. `_resample_linear` interpolates across `linspace(0, N-1)`,
  so its step depends on the total length and it cannot be used on a stream; the
  streaming one holds a fixed ratio and carries the unconsumed input samples and
  fractional read position between calls.
- `tests/test_tts_stream.py`, covering both fixes plus the resampler: chunked
  output is seam-free at any chunk size, a 220 Hz tone stays 220 Hz, the first
  frame is available from the first network chunk, a stop mid-stream abandons
  the response rather than draining it, and Piper audio rendered during a
  barge-in is discarded rather than played. Wired into CI.

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
