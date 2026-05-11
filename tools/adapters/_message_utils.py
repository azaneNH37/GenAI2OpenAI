import json
import logging

logger = logging.getLogger(__name__)


_IMAGE_PART_TYPES = ("image_url", "input_image")
_TEXT_PART_TYPES = ("text", "input_text")


def _normalize_content(content):
    if not isinstance(content, list):
        return content

    has_image = any(
        isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES
        for p in content
    )
    if has_image:
        kept = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in _IMAGE_PART_TYPES:
                kept.append(part)
            elif ptype in _TEXT_PART_TYPES:
                text = part.get("text", "")
                if text:
                    kept.append({"type": "text", "text": str(text)})
        return kept

    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def normalize_messages(messages):
    normalized = []
    for index, msg in enumerate(messages):
        role = msg.get("role")
        updated = dict(msg)
        updated["content"] = _normalize_content(msg.get("content"))

        if role == "tool":
            if not updated.get("tool_call_id"):
                updated["tool_call_id"] = f"unknown_{index}"

        if role == "assistant" and updated.get("tool_calls"):
            tool_calls = []
            for tc in updated.get("tool_calls", []):
                tc_copy = dict(tc)
                func = dict(tc.get("function", {}))
                args = func.get("arguments", "{}")
                if isinstance(args, dict):
                    func["arguments"] = json.dumps(args, ensure_ascii=False)
                else:
                    try:
                        json.loads(args)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning("Invalid tool_call arguments; using empty object")
                        func["arguments"] = json.dumps({})
                tc_copy["function"] = func
                tool_calls.append(tc_copy)
            updated["tool_calls"] = tool_calls

        normalized.append(updated)

    return normalized
