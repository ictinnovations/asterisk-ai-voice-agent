"""
LLM: Anthropic Claude, streamed, with tool_use.

Per-call instance: takes the persona's system prompt + tool specs, accumulates
caller turns, streams Claude's reply token-by-token, and surfaces tool_use
blocks so the orchestrator can execute them (via your webhook) and feed the
result back.

Swapping providers: keep the stream_reply() event shape ({'kind': ...}) and the
orchestrator doesn't care what's underneath.

Derived from ICTContact (https://www.ictcontact.com).
ICT Innovations (https://www.ictinnovations.com) - ICT Vision (https://ict.vision)
Author: Tahir Almas. MIT licensed.
"""

import json
import logging
from typing import AsyncIterator, List, Dict, Optional

from anthropic import AsyncAnthropic

log = logging.getLogger("ai.llm")

DEFAULT_MAX_TOKENS = 512

_PHP_TYPE_TO_JSON = {"string": "string", "integer": "integer",
                     "number": "number", "boolean": "boolean"}


def _to_anthropic_tools(specs: Dict[str, dict], enabled: List[str]) -> List[dict]:
    """Convert a spec dict { name: { description, parameters: {...} } } into the
    Anthropic SDK tool format, for the tools named in `enabled`."""
    out = []
    for name in enabled:
        spec = specs.get(name)
        if not spec:
            continue
        props = {}
        required = []
        for pname, p in (spec.get("parameters") or {}).items():
            props[pname] = {
                "type": _PHP_TYPE_TO_JSON.get(p.get("type", "string"), "string"),
                "description": p.get("description", ""),
            }
            # Treat every declared param as required for v1 - Claude is less
            # likely to hallucinate args this way.
            required.append(pname)
        out.append({
            "name": name,
            "description": spec.get("description", ""),
            "input_schema": {"type": "object", "properties": props, "required": required},
        })
    return out


class LLM:
    def __init__(self, provider: str, model: str, temperature: float,
                 system_prompt: str,
                 tool_specs: Optional[Dict[str, dict]] = None,
                 tools_enabled: Optional[List[str]] = None,
                 api_key: Optional[str] = None):
        self.provider      = provider
        self.model         = model or "claude-sonnet-4-6"
        self.temperature   = float(temperature) if temperature is not None else 0.4
        self.system_prompt = system_prompt or ""
        self.tool_specs    = tool_specs or {}
        self.tools_enabled = tools_enabled or []
        self.api_key       = api_key
        self.history: List[Dict] = []
        self._client: Optional[AsyncAnthropic] = None
        self._anth_tools = _to_anthropic_tools(self.tool_specs, self.tools_enabled)

    async def start(self) -> None:
        if self.provider != "anthropic":
            log.warning("llm provider=%s not implemented; using anthropic", self.provider)
        if not self.api_key:
            raise RuntimeError("Anthropic API key not configured (providers.anthropic.api_key)")
        self._client = AsyncAnthropic(api_key=self.api_key)
        log.info("LLM ready: model=%s temp=%s tools=%d",
                 self.model, self.temperature, len(self._anth_tools))

    async def add_user(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # A barge-in can leave the last history entry as role=user. Merge instead
        # of appending a second consecutive user message.
        if self.history and self.history[-1]["role"] == "user":
            prev = self.history[-1]
            if isinstance(prev["content"], str):
                prev["content"] = [{"type": "text", "text": prev["content"]}]
            prev["content"].append({"type": "text", "text": text})
            return
        self.history.append({"role": "user", "content": text})

    async def add_assistant_content(self, content) -> None:
        """Append an assistant message (str or block list) to history verbatim."""
        if isinstance(content, str):
            content = content.strip()
            if not content:
                return
        self.history.append({"role": "assistant", "content": content})

    async def add_tool_result(self, tool_use_id: str, result_obj) -> None:
        """Feed a tool's result back to Claude so it can continue."""
        result_str = json.dumps(result_obj) if not isinstance(result_obj, str) else result_obj
        self.history.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result_str}],
        })

    async def stream_reply(self) -> AsyncIterator[dict]:
        """
        Stream the assistant turn. Yields a mix of:
          {'kind': 'text', 'text': str}                 - partial text delta
          {'kind': 'tool', 'name': str, 'args': dict, 'id': str}  - run a tool
          {'kind': 'end',  'stop_reason': str, 'assistant_blocks': list}

        After 'end', the orchestrator runs any tool calls, calls
        add_tool_result(...) per tool, and calls stream_reply() again for
        Claude's next turn.
        """
        if self._client is None:
            await self.start()

        kwargs = dict(
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=self.history,
        )
        if self._anth_tools:
            kwargs["tools"] = self._anth_tools

        assistant_blocks: List[Dict] = []
        cur_text = ""
        cur_tool_id: Optional[str] = None
        cur_tool_name: Optional[str] = None
        cur_tool_input_json = ""
        stop_reason = "end_turn"

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    t = getattr(event, "type", None)
                    if t == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            cur_tool_id = block.id
                            cur_tool_name = block.name
                            cur_tool_input_json = ""
                        elif block.type == "text":
                            cur_text = ""
                    elif t == "content_block_delta":
                        d = event.delta
                        if getattr(d, "type", "") == "text_delta":
                            cur_text += d.text
                            yield {"kind": "text", "text": d.text}
                        elif getattr(d, "type", "") == "input_json_delta":
                            cur_tool_input_json += d.partial_json
                    elif t == "content_block_stop":
                        if cur_tool_id:
                            try:
                                args = json.loads(cur_tool_input_json) if cur_tool_input_json else {}
                            except Exception:
                                args = {}
                            assistant_blocks.append({
                                "type": "tool_use", "id": cur_tool_id,
                                "name": cur_tool_name, "input": args,
                            })
                            yield {"kind": "tool", "name": cur_tool_name, "args": args, "id": cur_tool_id}
                            cur_tool_id = None
                            cur_tool_name = None
                            cur_tool_input_json = ""
                        elif cur_text:
                            assistant_blocks.append({"type": "text", "text": cur_text})
                            cur_text = ""
                    elif t == "message_delta":
                        if getattr(event, "delta", None) and getattr(event.delta, "stop_reason", None):
                            stop_reason = event.delta.stop_reason
        except Exception as e:
            log.exception("Anthropic stream error: %s", e)
            msg = "I'm sorry, I had a problem. Could you say that again?"
            yield {"kind": "text", "text": msg}
            assistant_blocks.append({"type": "text", "text": msg})
            stop_reason = "end_turn"

        # Append the full assistant message - required for the tool_result loop.
        if assistant_blocks:
            await self.add_assistant_content(assistant_blocks)
        yield {"kind": "end", "stop_reason": stop_reason, "assistant_blocks": assistant_blocks}

    async def close(self) -> None:
        if self._client:
            await self._client.close()
