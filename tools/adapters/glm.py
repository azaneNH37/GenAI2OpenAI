import json
import re
import uuid

from tools.adapters._inject_shared import inject_with_renderers
from tools.adapters.base import ToolAdapter
from tools.prompts import normalize_messages
from tools.parsing import (
    ToolCallParseResult,
    _canonical_tool_name,
    _coerce_arguments,
    _find_tool_call_blocks,
    _try_load_json,
    _validate_arguments,
    normalize_whitespace_around_tags,
    strip_think_blocks,
    unwrap_markdown_fences,
)

TOOL_SYSTEM_PROMPT = """\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, output the function name and arguments in this format:
<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value><arg_key>{{arg-key-2}}</arg_key><arg_value>{{arg-value-2}}</arg_value>...</tool_call>
"""

TOOL_CHOICE_REQUIRED_PROMPT = "\nFor this turn, you must call at least one tool using a <tool_call> block."
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nFor this turn, you must call the tool named "{name}" using a <tool_call> block.'
)
TOOL_CHOICE_NONE_PROMPT = "For this turn, do not call any tool or emit <tool_call> tags."


def _format_tool_definitions(tools):
    lines = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        lines.append(json.dumps(tool.get("function", {}), ensure_ascii=False))
    return "\n".join(lines)


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _render_tool_call_message(msg):
    parts = []
    if msg.get("content"):
        parts.append(str(msg["content"]))
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        arguments = _safe_json_loads(func.get("arguments", "{}"))
        arg_parts = []
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                arg_parts.append(
                    f"<arg_key>{key}</arg_key><arg_value>{rendered}</arg_value>"
                )
        parts.append(f"<tool_call>{func.get('name', '')}{''.join(arg_parts)}</tool_call>")
    return "\n\n".join(p for p in parts if p).strip()


def _normalize_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _render_tool_results(tool_messages):
    return "<|observation|>" + "".join(
        f"<tool_response>{_normalize_content(msg.get('content'))}</tool_response>"
        for msg in tool_messages
    )


def _parse_glm_tool_call_body(raw):
    raw = raw.strip()

    first_key_pos = raw.find("<arg_key>")
    if first_key_pos > 0:
        name = raw[:first_key_pos].strip()
        if name:
            arguments = {}
            for key, value in re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                raw,
                re.DOTALL,
            ):
                arguments[key.strip()] = value.strip()
            return {"name": name, "arguments": arguments}, "glm"

    call = _try_load_json(raw)
    if isinstance(call, dict) and "name" in call:
        return call, "json"

    name_m = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    args_m = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            parsed_args = _try_load_json(args_str)
            if isinstance(parsed_args, dict):
                arguments = parsed_args
            else:
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}, "xml"

    return None, "parse_failed"


class GlmAdapter(ToolAdapter):
    def inject(self, messages, tools, tool_choice=None):
        messages = normalize_messages(messages)

        if tool_choice == "none":
            tool_prompt = TOOL_CHOICE_NONE_PROMPT
        else:
            tool_defs = _format_tool_definitions(tools)
            tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

            if tool_choice == "required":
                tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                name = tool_choice["function"]["name"]
                tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

        return inject_with_renderers(
            messages,
            tool_prompt,
            _render_tool_call_message,
            _render_tool_results,
        )

    def extract_tool_calls(self, content, tools=None):
        cleaned = strip_think_blocks(content or "")
        cleaned = unwrap_markdown_fences(cleaned)
        cleaned = normalize_whitespace_around_tags(cleaned)

        blocks = _find_tool_call_blocks(cleaned)
        missing_close = False
        if not blocks:
            open_tag = "<tool_call>"
            start = cleaned.find(open_tag)
            if start != -1:
                body = cleaned[start + len(open_tag):]
                blocks = [(body, start, len(cleaned))]
                missing_close = True
            else:
                return ToolCallParseResult(None, content, [], cleaned)

        tool_calls = []
        parse_errors = []

        for i, (body, start, end) in enumerate(blocks):
            call, method = _parse_glm_tool_call_body(body)
            if not call or "name" not in call:
                parse_errors.append(f"tool_call[{i}] parse_failed")
                continue

            canonical_name = _canonical_tool_name(call.get("name", ""), tools)
            arguments = call.get("arguments", {})

            coerced_args = _coerce_arguments(arguments, canonical_name, tools)
            missing = _validate_arguments(coerced_args, canonical_name, tools)
            if missing:
                parse_errors.append(
                    f"tool_call[{i}] missing required: {', '.join(missing)}"
                )

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": canonical_name,
                    "arguments": json.dumps(coerced_args, ensure_ascii=False),
                },
            })

            if method in ("json", "xml"):
                parse_errors.append(f"tool_call[{i}] parsed via fallback: {method}")
            if missing_close:
                parse_errors.append(f"tool_call[{i}] missing closing tag")

        if not tool_calls:
            return ToolCallParseResult(None, content, parse_errors, cleaned)

        remaining_parts = []
        cursor = 0
        for _, start, end in blocks:
            if cursor < start:
                remaining_parts.append(cleaned[cursor:start])
            cursor = end
        if cursor < len(cleaned):
            remaining_parts.append(cleaned[cursor:])

        remaining_text = "".join(remaining_parts).strip() or None
        return ToolCallParseResult(tool_calls, remaining_text, parse_errors, cleaned)
