"""MiniMax 工具适配器。

MiniMax M1 / M2 系列在 GenAI 平台上使用 ``<minimax:tool_call>`` 包裹的 XML invoke
风格，区别于 GLM 的扁平 ``<arg_key>`` 标签和 DeepSeek 的 DSML token。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tools.adapters.base import ToolAdapter
from tools.adapters._message_utils import normalize_messages
from tools.parsing import ToolCallParseResult, canonical_tool_name


_TEMPLATE = """\
# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
{tool_definitions}
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
</minimax:tool_call>"""

_REQUIRED_SUFFIX = (
    "\nFor this turn, you must call at least one tool using a <minimax:tool_call> block."
)
_SPECIFIC_SUFFIX = (
    '\nFor this turn, you must call the tool named "{name}" using a <minimax:tool_call> block.'
)
_NONE_SUFFIX = "\nFor this turn, do not call any tool or emit tool call tags."


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _decode_param_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _tool_choice_is_none(tool_choice: Any) -> bool:
    if tool_choice == "none":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


class MinimaxAdapter(ToolAdapter):
    open_tags: tuple[str, ...] = ("<minimax:tool_call>", "<tool_call>")

    def inject(self, messages, tools, tool_choice=None):
        messages = normalize_messages(messages)
        tool_prompt = self._render_prompt(tools, tool_choice)
        allow_more = not _tool_choice_is_none(tool_choice)

        new_messages = []
        has_system = False
        index = 0

        while index < len(messages):
            msg = messages[index]
            role = msg.get("role")

            if role == "system":
                new_messages.append({
                    "role": "system",
                    "content": (msg.get("content", "") or "") + "\n\n" + tool_prompt,
                })
                has_system = True
                index += 1
                continue

            if role == "assistant" and msg.get("tool_calls"):
                new_messages.append({
                    "role": "assistant",
                    "content": self._render_tool_call_message(msg),
                })
                index += 1
                continue

            if role == "tool":
                tool_messages = []
                while index < len(messages) and messages[index].get("role") == "tool":
                    tool_messages.append(messages[index])
                    index += 1
                new_messages.append({
                    "role": "user",
                    "content": self._render_tool_results(tool_messages, allow_more),
                })
                continue

            new_messages.append(msg)
            index += 1

        if not has_system:
            new_messages.insert(0, {"role": "system", "content": tool_prompt})
        return new_messages

    def _render_prompt(self, tools, tool_choice):
        defs = []
        for tool in tools or []:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            if func:
                defs.append(f"<tool>{json.dumps(func, ensure_ascii=False)}</tool>")

        prompt = _TEMPLATE.format(tool_definitions="\n".join(defs))
        if tool_choice == "required":
            prompt += _REQUIRED_SUFFIX
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            prompt += _SPECIFIC_SUFFIX.format(name=tool_choice["function"]["name"])
        elif _tool_choice_is_none(tool_choice):
            prompt += _NONE_SUFFIX
        return prompt

    def _render_tool_call_message(self, msg):
        parts = []
        if msg.get("content"):
            parts.append(str(msg["content"]))
        invocations = []
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            args = _safe_json_loads(func.get("arguments", "{}"))
            param_lines = []
            if isinstance(args, dict):
                for k, v in args.items():
                    rendered = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    param_lines.append(f'<parameter name="{k}">{rendered}</parameter>')
            invocations.append("\n".join([
                f'<invoke name="{func.get("name", "")}">',
                *param_lines,
                "</invoke>",
            ]))
        if invocations:
            parts.append("\n".join(["<minimax:tool_call>", *invocations, "</minimax:tool_call>"]))
        return "\n\n".join(p for p in parts if p).strip()

    def _render_tool_results(self, tool_messages, allow_more: bool) -> str:
        body = "\n".join(
            f"<response>{_normalize_content(m.get('content'))}</response>"
            for m in tool_messages
        )
        if allow_more:
            return body + "\nUse these tool results to answer the user. Only call another tool if the current result is genuinely insufficient."
        return body + "\nAnswer the user normally using these tool results. Do not call any tool."

    # ---------- extract ----------

    def extract_tool_calls(self, content, tools=None):
        if not content:
            return ToolCallParseResult(None, content, [], content or "")

        match = re.search(
            r"<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>",
            content,
            re.DOTALL,
        )
        if not match:
            # 兼容直接发出 <invoke> 块（无外层包裹）
            if "<invoke" in content and "</invoke>" in content:
                block = content
                start, end = 0, len(content)
            else:
                # 回退 generic
                from tools.adapters.generic import GenericAdapter
                return GenericAdapter().extract_tool_calls(content, tools=tools)
        else:
            block = match.group(1)
            start, end = match.start(), match.end()

        invocations = re.findall(
            r'<invoke name="(.*?)">(.*?)</invoke>',
            block,
            re.DOTALL,
        )
        if not invocations:
            from tools.adapters.generic import GenericAdapter
            return GenericAdapter().extract_tool_calls(content, tools=tools)

        tool_calls = []
        errors = []
        for name, body in invocations:
            args = {}
            for p_name, raw_val in re.findall(
                r'<parameter name="(.*?)">(.*?)</parameter>',
                body,
                re.DOTALL,
            ):
                args[p_name] = _decode_param_value(raw_val)
            canonical = canonical_tool_name(name, tools)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": canonical,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        remaining = (content[:start] + content[end:]).strip() or None
        return ToolCallParseResult(tool_calls, remaining, errors, content)
