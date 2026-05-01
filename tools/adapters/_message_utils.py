import json
import logging

logger = logging.getLogger(__name__)


def _normalize_content(content):
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return content


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
