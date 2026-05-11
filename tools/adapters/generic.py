import json
import re
import uuid
from typing import Any

from tools.adapters._inject_shared import inject_with_renderers
from tools.adapters.base import ToolAdapter
from tools.adapters._message_utils import normalize_messages
from tools.parsing import (
    ToolCallParseResult,
    canonical_tool_name,
    coerce_arguments,
    find_tool_call_blocks,
    parse_tool_call_body,
    try_load_json,
    validate_arguments,
    normalize_whitespace_around_tags,
    strip_think_blocks,
    unwrap_markdown_fences,
)

TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

When you need to call a tool, output EXACTLY one <tool_call> block per invocation:

<tool_call>
{{"name": "<tool-name>", "arguments": {{<arguments-as-json-object>}}}}
</tool_call>

Constraints:
1. The JSON inside <tool_call> MUST be a single valid JSON object with keys "name" (string)
   and "arguments" (object).
2. Do NOT wrap <tool_call> in markdown code fences.
3. Multiple parallel calls: emit multiple <tool_call> blocks in sequence.
4. After receiving <tool_result> blocks, reason and give a final plain-text answer.
5. If no tool is needed, respond in plain text — do NOT emit any <tool_call> tag.
"""

TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = '\nYou MUST call the tool named "{name}" and no other tool.'
TOOL_CHOICE_NONE_PROMPT = "Do NOT call any tool. Respond in plain text only."


def _format_tool_definitions(tools):
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f"<tool_definition>\n"
            f"  <name>{func['name']}</name>\n"
            f"  <description>{func.get('description', '')}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool_definition>"
        )
    return "\n".join(definitions)


def _render_tool_call_message(msg):
    parts = []
    if msg.get("content"):
        parts.append(str(msg["content"]))
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        call_obj = {
            "name": func.get("name", ""),
            "arguments": _safe_json_loads(func.get("arguments", "{}")),
        }
        parts.append(
            f"<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"
        )
    return "\n".join(p for p in parts if p).strip()


def _render_tool_results(tool_messages):
    blocks = []
    for msg in tool_messages:
        tool_call_id = msg.get("tool_call_id", "unknown")
        blocks.append(
            "<tool_result>\n"
            f"  <tool_call_id>{tool_call_id}</tool_call_id>\n"
            f"  <content>{msg.get('content', '')}</content>\n"
            "</tool_result>"
        )
    return "<tool_results>\nUse these tool results to answer the user. Only call another tool if genuinely insufficient.\n" + "\n".join(blocks) + "\n</tool_results>"


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _tool_schema_map(tools):
    schema_map = {}
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name")
        if not name:
            continue
        schema_map[name] = func.get("parameters", {})
    return schema_map


def _skip_ws(text, start):
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _parse_lenient_json_string(text, start):
    if text[start] != '"':
        raise ValueError("String must start with quote")

    index = start + 1
    buffer = []
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            buffer.append(text[index + 1])
            index += 2
            continue
        if char == '"':
            lookahead = _skip_ws(text, index + 1)
            if lookahead >= len(text) or text[lookahead] in ",]}:":
                return "".join(buffer), index + 1
        buffer.append(char)
        index += 1

    raise ValueError("String not terminated")


def _parse_lenient_json_value(text, start):
    if start >= len(text):
        raise ValueError("Missing value")

    char = text[start]
    if char == '"':
        return _parse_lenient_json_string(text, start)
    if char == "{":
        return _parse_lenient_json_object(text, start)
    if char == "[":
        return _parse_lenient_json_array(text, start)

    end = start
    while end < len(text) and text[end] not in ",]}":
        end += 1

    raw = text[start:end].strip()
    if not raw:
        raise ValueError("Empty scalar")

    if raw == "true":
        return True, end
    if raw == "false":
        return False, end
    if raw == "null":
        return None, end

    try:
        return int(raw), end
    except ValueError:
        try:
            return float(raw), end
        except ValueError:
            return raw, end


def _parse_lenient_object_key(text, start):
    if text[start] == '"':
        return _parse_lenient_json_string(text, start)
    end = start
    while end < len(text) and (text[end].isalnum() or text[end] in '_-'):
        end += 1
    if end == start:
        raise ValueError("Empty object key")
    return text[start:end], end


def _parse_lenient_json_object(text, start):
    if text[start] != "{":
        raise ValueError("Object must start with '{'")

    index = start + 1
    result = {}

    while index < len(text):
        index = _skip_ws(text, index)
        if index >= len(text):
            raise ValueError("Unexpected end of object")
        if text[index] == "}":
            return result, index + 1

        key, index = _parse_lenient_object_key(text, index)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise ValueError("Missing ':' after object key")

        index += 1
        index = _skip_ws(text, index)
        value, index = _parse_lenient_json_value(text, index)
        result[key] = value

        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return result, index + 1
        raise ValueError("Invalid object separator")

    raise ValueError("Object not terminated")


def _parse_lenient_json_array(text, start):
    if text[start] != "[":
        raise ValueError("Array must start with '['")

    index = start + 1
    result = []

    while index < len(text):
        index = _skip_ws(text, index)
        if index >= len(text):
            raise ValueError("Unexpected end of array")
        if text[index] == "]":
            return result, index + 1

        value, index = _parse_lenient_json_value(text, index)
        result.append(value)

        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return result, index + 1
        raise ValueError("Invalid array separator")

    raise ValueError("Array not terminated")


def _coerce_scalar(value, expected_type):
    if expected_type == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if expected_type == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if expected_type == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _coerce_with_schema(value, schema):
    if not isinstance(value, dict):
        return value

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return value

    coerced = {}
    for key, raw_value in value.items():
        prop_schema = properties.get(key, {})
        prop_type = prop_schema.get("type")
        coerced[key] = _coerce_scalar(raw_value, prop_type)
    return coerced


def _extract_arguments(raw, schema):
    xml_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", raw, re.DOTALL)
    if xml_match:
        raw_args = xml_match.group(1).strip()
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

    args_match = re.search(r'"arguments"\s*:\s*', raw)
    if not args_match:
        return {}

    idx = args_match.end()
    while idx < len(raw) and raw[idx].isspace():
        idx += 1

    if idx >= len(raw):
        return {}

    if raw[idx] != "{":
        return {}

    try:
        parsed, _ = _parse_lenient_json_object(raw, idx)
        return _coerce_with_schema(parsed, schema)
    except ValueError:
        return None


def _repair_tool_call_body(raw, tool_schemas):
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if not name_match:
        name_match = re.search(r"<name>\s*(.*?)\s*</name>", raw, re.DOTALL)
    if not name_match:
        return None

    name = name_match.group(1).strip()
    arguments = _extract_arguments(raw, tool_schemas.get(name, {}))
    if arguments is None:
        return None
    return {"name": name, "arguments": arguments}


class GenericAdapter(ToolAdapter):
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

        def _normalize_attr(match):
            name = match.group(1)
            body = match.group(2).strip()
            if body.startswith('"arguments"'):
                return f'<tool_call>{{"name": "{name}", {body}}}</tool_call>'
            if body.startswith("{"):
                return f'<tool_call>{{"name": "{name}", "arguments": {body}}}</tool_call>'
            return f'<tool_call>{{"name": "{name}", "arguments": {{{body}}}}}</tool_call>'

        cleaned = re.sub(
            r'<tool_call\s+name="([^"]+)"\s*>(.*?)</tool_call>',
            _normalize_attr,
            cleaned,
            flags=re.DOTALL,
        )

        cleaned = normalize_whitespace_around_tags(cleaned)

        blocks = find_tool_call_blocks(cleaned)
        missing_close = False
        if not blocks:
            start = cleaned.find("<tool_call>")
            if start != -1:
                body = cleaned[start + len("<tool_call>"):]
                blocks = [(body, start, len(cleaned))]
                missing_close = True
            else:
                return ToolCallParseResult(None, content, [], cleaned)

        tool_schemas = _tool_schema_map(tools or [])
        tool_calls = []
        parse_errors = []

        for i, (body, start, end) in enumerate(blocks):
            call, method = parse_tool_call_body(body)
            if call and "name" in call:
                canonical_name = canonical_tool_name(call.get("name", ""), tools)
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    parsed_args = try_load_json(arguments)
                    if isinstance(parsed_args, dict):
                        arguments = parsed_args

                if method == "lenient" and (not isinstance(arguments, dict) or not arguments):
                    repaired = _repair_tool_call_body(body, tool_schemas)
                    if repaired and repaired.get("name"):
                        canonical_name = canonical_tool_name(repaired["name"], tools)
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": canonical_name,
                                "arguments": json.dumps(repaired["arguments"], ensure_ascii=False),
                            },
                        })
                        parse_errors.append(f"tool_call[{i}] parsed via repair mode (lenient rejected)")
                        continue

                coerced_args = coerce_arguments(arguments, canonical_name, tools)
                missing = validate_arguments(coerced_args, canonical_name, tools)
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

                if method == "lenient":
                    parse_errors.append(f"tool_call[{i}] parsed via lenient mode")
                continue

            repaired = _repair_tool_call_body(body, tool_schemas)
            if repaired and repaired.get("name"):
                canonical_name = canonical_tool_name(repaired["name"], tools)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": canonical_name,
                        "arguments": json.dumps(repaired["arguments"], ensure_ascii=False),
                    },
                })
                parse_errors.append(f"tool_call[{i}] parsed via repair mode")
                continue

            parse_errors.append(f"tool_call[{i}] parse_failed")

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
