import json
import logging


logger = logging.getLogger(__name__)

TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

## Rules for tool calls

When you decide to call a tool, output EXACTLY ONE <tool_call> block per tool invocation.
Format:

<tool_call>
{{"name": "<tool-name>", "arguments": {{<arguments-as-json-object>}}}}
</tool_call>

Constraints:
1. The JSON inside <tool_call> MUST be a single valid JSON object with keys "name" (string) and "arguments" (object).
2. Do NOT wrap <tool_call> in markdown code fences.
3. You may emit multiple <tool_call> blocks in one response for parallel calls.
4. After receiving <tool_result> blocks, continue reasoning or give a final plain-text answer.
5. If you do not need any tool, respond in plain text only - do NOT emit any <tool_call> tag.
"""

TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" and no other tool.'
)
TOOL_CHOICE_NONE_PROMPT = "Do NOT call any tool. Respond in plain text only."


def format_tool_definitions(tools):
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


def inject_tool_prompt(messages, tools, tool_choice=None):
    from tools.adapters.generic import GenericAdapter

    return GenericAdapter().inject(messages, tools, tool_choice)
