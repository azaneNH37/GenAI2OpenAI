import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)


def strip_think_blocks(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


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


def _parse_tool_call_body(raw):
    raw = raw.strip()

    call = _try_load_json(raw)
    if isinstance(call, dict) and "name" in call:
        if isinstance(call.get("arguments"), str):
            parsed_args = _try_load_json(call["arguments"])
            if isinstance(parsed_args, dict):
                call["arguments"] = parsed_args
        return call

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
        return {"name": name, "arguments": arguments}

    return None


def extract_tool_calls(content):
    cleaned = strip_think_blocks(content)

    cleaned = re.sub(
        r'```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```',
        r'\1',
        cleaned,
        flags=re.DOTALL
    )

    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, cleaned, re.DOTALL)

    if not matches:
        logger.debug("No <tool_call> tags found in content (%d chars): %s",
                      len(content), content[:500])
        return None, content

    logger.debug("Found %d <tool_call> match(es)", len(matches))

    tool_calls = []
    for i, match in enumerate(matches):
        call = _parse_tool_call_body(match)
        if call:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments", {}),
                        ensure_ascii=False
                    )
                }
            })
        else:
            logger.warning("Failed to parse tool_call[%d] — raw: %s", i, match[:300])
            continue

    if not tool_calls:
        return None, content

    remaining = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL).strip()
    return tool_calls, remaining or None


def _tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0
