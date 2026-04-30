import json
import logging
import re
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCallParseResult:
    tool_calls: list | None
    remaining_text: str | None
    parse_errors: list
    raw_content: str


def strip_think_blocks(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)


def unwrap_markdown_fences(content):
    return re.sub(
        r'```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```',
        r'\1',
        content,
        flags=re.DOTALL
    )


def normalize_whitespace_around_tags(content):
    def _cleanup(match):
        body = match.group(1).strip()
        return f"<tool_call>\n{body}\n</tool_call>"

    return re.sub(r'<tool_call>\s*(.*?)\s*</tool_call>', _cleanup, content, flags=re.DOTALL)


def _escape_invalid_backslashes(text):
    out = []
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue

        if escape:
            out.append(ch)
            escape = False
            continue

        if ch == '\\':
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                out.append(ch)
                escape = True
            else:
                out.append('\\\\')
            continue

        if ch == '"':
            in_string = False
            out.append(ch)
            continue

        out.append(ch)

    return "".join(out)


def _try_load_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    repaired = _escape_invalid_backslashes(raw)
    if repaired != raw:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _find_first_json_object(raw):
    in_string = False
    escape = False
    depth = 0
    start = None

    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return raw[start:i + 1]

    return None


def _parse_lenient_kv(raw):
    pairs = re.findall(r'"?([a-zA-Z0-9_\-]+)"?\s*:\s*([^,\n}]+)', raw)
    if not pairs:
        return None

    data = {}
    for key, value in pairs:
        value = value.strip().strip('"')
        data[key] = value
    return data


def _parse_tool_call_body(raw):
    raw = raw.strip()

    call = _try_load_json(raw)
    if isinstance(call, dict) and "name" in call:
        return call, None

    name_m = re.search(r'<name>\s*(.*?)\s*</name>', raw, re.DOTALL)
    args_m = re.search(r'<arguments>\s*(.*?)\s*</arguments>', raw, re.DOTALL)
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
        return {"name": name, "arguments": arguments}, None

    raw_obj = _find_first_json_object(raw)
    if raw_obj:
        call = _try_load_json(raw_obj)
        if isinstance(call, dict) and "name" in call:
            return call, None

    kv = _parse_lenient_kv(raw)
    if isinstance(kv, dict) and "name" in kv:
        return kv, "lenient"

    return None, "parse_failed"


def _canonical_tool_name(name, tools):
    if not tools:
        return name
    known = {
        tool["function"]["name"]: tool["function"]["name"]
        for tool in tools if tool.get("type") == "function"
    }
    if name in known:
        return name
    lower_map = {key.lower(): value for key, value in known.items()}
    return lower_map.get(name.lower(), name)


def _parse_typed_value(value, expected_type):
    if expected_type == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if expected_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if expected_type == "boolean":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
        return value
    if expected_type in ("array", "object") and isinstance(value, str):
        parsed = _try_load_json(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    return value


def _coerce_arguments(arguments, tool_name, tools):
    if not tools:
        return arguments
    if not isinstance(arguments, dict):
        return arguments

    tool_schema = None
    for tool in tools:
        if tool.get("type") == "function" and tool.get("function", {}).get("name") == tool_name:
            tool_schema = tool.get("function", {}).get("parameters", {})
            break

    if not tool_schema:
        return arguments

    properties = tool_schema.get("properties", {})
    coerced = dict(arguments)
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        if expected:
            coerced[key] = _parse_typed_value(value, expected)
    return coerced


def _validate_arguments(arguments, tool_name, tools):
    if not tools:
        return []
    if not isinstance(arguments, dict):
        return []

    tool_schema = None
    for tool in tools:
        if tool.get("type") == "function" and tool.get("function", {}).get("name") == tool_name:
            tool_schema = tool.get("function", {}).get("parameters", {})
            break

    if not tool_schema:
        return []

    required = tool_schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    return missing


def _find_tool_call_blocks(content):
    blocks = []
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL):
        blocks.append((match.group(1), match.start(), match.end()))
    return blocks


def extract_tool_calls(content, tools=None):
    from tools.adapters.generic import GenericAdapter

    return GenericAdapter().extract_tool_calls(content, tools=tools)


def _tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0
