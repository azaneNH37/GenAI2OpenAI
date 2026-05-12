"""DeepSeek DSML 工具适配器。

DeepSeek 在 GenAI 平台上对 tool calling 使用一种基于 DSML（DeepSeek Markup Language）
的 XML 风格协议。本文件提供两个变体：

- ``DEEPSEEK_LEGACY``: 旧版 DeepSeek（v3 / v3.1），使用 ``<｜DSML｜function_calls>``
- ``DEEPSEEK_V4``: 新版 DeepSeek v4（flash / pro），使用 ``<｜DSML｜tool_calls>``

参数序列化形式相同，均为 ``<｜DSML｜parameter name="..." string="true|false">value</...>``。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tools.adapters.base import ToolAdapter
from tools.adapters._message_utils import normalize_messages
from tools.parsing import ToolCallParseResult, canonical_tool_name


DSML_TOKEN = "｜DSML｜"
DSML_TOOL_CALLS_START = "<｜DSML｜tool_calls>"
DSML_TOOL_CALLS_END = "</｜DSML｜tool_calls>"
DSML_FUNCTION_CALLS_START = "<｜DSML｜function_calls>"
DSML_FUNCTION_CALLS_END = "</｜DSML｜function_calls>"

VARIANT_LEGACY = "legacy"
VARIANT_V4 = "v4"


_V4_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

_LEGACY_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.

You can invoke functions by writing a "<｜DSML｜function_calls>" block like the following as part of your reply:
<｜DSML｜function_calls>
<｜DSML｜invoke name="$FUNCTION_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
</｜DSML｜function_calls>

String and scalar parameters should be specified as-is without any escaping or quotes.
Lists and objects should use JSON format.
The "string" attribute must be "true" for string parameters and "false" for numbers, booleans, arrays, and objects.

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""

_REQUIRED_SUFFIX_V4 = (
    "\nFor this turn, a plain-text-only answer is invalid. "
    "You must emit a <｜DSML｜tool_calls> block."
)
_REQUIRED_SUFFIX_LEGACY = (
    "\nFor this turn, a plain-text-only answer is invalid. "
    "You must emit a <｜DSML｜function_calls> block."
)
_NONE_SUFFIX = (
    "\nFor this turn, do not emit DSML tool_calls/function_calls or <tool_call> tags."
)


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


def _decode_param(raw_value: str, is_string: bool) -> Any:
    value = raw_value.strip()
    if is_string:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _render_param(key: str, value: Any) -> str:
    is_string = isinstance(value, str)
    rendered = value if is_string else json.dumps(value, ensure_ascii=False)
    flag = "true" if is_string else "false"
    return (
        f'<{DSML_TOKEN}parameter name="{key}" string="{flag}">'
        f"{rendered}</{DSML_TOKEN}parameter>"
    )


def _tool_choice_is_none(tool_choice: Any) -> bool:
    if tool_choice == "none":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


class DeepSeekAdapter(ToolAdapter):
    open_tags: tuple[str, ...] = (DSML_TOOL_CALLS_START, DSML_FUNCTION_CALLS_START, "<tool_call>")

    def __init__(self, variant: str = VARIANT_V4):
        if variant not in (VARIANT_LEGACY, VARIANT_V4):
            raise ValueError(f"Unknown deepseek variant: {variant}")
        self.variant = variant
        self._is_legacy = variant == VARIANT_LEGACY
        self._block_name = "function_calls" if self._is_legacy else "tool_calls"
        self._start_tag = DSML_FUNCTION_CALLS_START if self._is_legacy else DSML_TOOL_CALLS_START
        self._end_tag = DSML_FUNCTION_CALLS_END if self._is_legacy else DSML_TOOL_CALLS_END

    # ---------- inject ----------

    def inject(self, messages, tools, tool_choice=None):
        messages = normalize_messages(messages)
        tool_prompt = self._render_prompt(tools, tool_choice)

        new_messages = []
        has_system = False
        index = 0
        allow_more = not _tool_choice_is_none(tool_choice)

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
                parts = []
                if msg.get("content"):
                    parts.append(str(msg["content"]))
                parts.append(self._render_tool_calls(msg["tool_calls"]))
                new_messages.append({
                    "role": "assistant",
                    "content": "\n\n".join(p for p in parts if p).strip(),
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
        schemas = []
        for tool in tools or []:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            if func:
                schemas.append(json.dumps(func, ensure_ascii=False))

        template = _LEGACY_TEMPLATE if self._is_legacy else _V4_TEMPLATE
        prompt = template.format(tool_schemas="\n".join(schemas))

        if tool_choice == "required":
            prompt += _REQUIRED_SUFFIX_LEGACY if self._is_legacy else _REQUIRED_SUFFIX_V4
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            name = tool_choice["function"]["name"]
            prompt += (
                f'\nFor this turn, you must emit a <{self._start_tag[1:-1]}> block '
                f'using the tool "{name}".'
            )
        elif _tool_choice_is_none(tool_choice):
            prompt += _NONE_SUFFIX

        return prompt

    def _render_tool_calls(self, tool_calls) -> str:
        invocations = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = _safe_json_loads(func.get("arguments", "{}"))
            param_lines = []
            if isinstance(args, dict):
                for k, v in args.items():
                    param_lines.append(_render_param(k, v))
            invocations.append("\n".join([
                f'<{DSML_TOKEN}invoke name="{name}">',
                *param_lines,
                f'</{DSML_TOKEN}invoke>',
            ]))

        return "\n".join([
            f'<{DSML_TOKEN}{self._block_name}>',
            *invocations,
            f'</{DSML_TOKEN}{self._block_name}>',
        ])

    def _render_tool_results(self, tool_messages, allow_more: bool) -> str:
        if not self._is_legacy:
            return "\n".join(
                f"<tool_result>{_normalize_content(m.get('content'))}</tool_result>"
                for m in tool_messages
            )
        lines = ["<function_results>"]
        if not allow_more:
            lines.append("Answer the user normally using these function results.")
        for m in tool_messages:
            lines.append(f"<tool_result>{_normalize_content(m.get('content'))}</tool_result>")
        lines.append("</function_results>")
        return "\n".join(lines)

    # ---------- extract ----------

    def extract_tool_calls(self, content, tools=None):
        if not content:
            return ToolCallParseResult(None, content, [], content or "")

        # 优先解析 DSML 块（同时兼容 v4 / legacy 两种标签）
        dsml_calls, dsml_remaining, dsml_errors = self._extract_dsml(content, tools)
        if dsml_calls:
            return ToolCallParseResult(dsml_calls, dsml_remaining, dsml_errors, content)

        # 回退到通用 <tool_call>{...}</tool_call> 解析
        from tools.adapters.generic import GenericAdapter
        return GenericAdapter().extract_tool_calls(content, tools=tools)

    def _extract_dsml(self, content: str, tools):
        pairs = [
            (DSML_TOOL_CALLS_START, DSML_TOOL_CALLS_END),
            (DSML_FUNCTION_CALLS_START, DSML_FUNCTION_CALLS_END),
        ]
        match = None
        for start, end in pairs:
            m = re.search(
                rf"{re.escape(start)}(.*?){re.escape(end)}",
                content,
                re.DOTALL,
            )
            if m:
                match = m
                break

        if not match:
            return None, content, []

        block = match.group(1)
        invocations = re.findall(
            r'<｜DSML｜invoke name="(.*?)">(.*?)</｜DSML｜invoke>',
            block,
            re.DOTALL,
        )
        if not invocations:
            return None, content, ["dsml block found but no invocations"]

        tool_calls = []
        errors = []
        for tool_name, raw_params in invocations:
            arguments = {}
            for p_name, is_str, raw_val in re.findall(
                r'<｜DSML｜parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜parameter>',
                raw_params,
                re.DOTALL,
            ):
                arguments[p_name] = _decode_param(raw_val, is_str == "true")

            canonical = canonical_tool_name(tool_name, tools)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": canonical,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })

        remaining = (content[: match.start()] + content[match.end():]).strip() or None
        return tool_calls, remaining, errors


DeepSeekV4Adapter = lambda: DeepSeekAdapter(VARIANT_V4)
DeepSeekLegacyAdapter = lambda: DeepSeekAdapter(VARIANT_LEGACY)
