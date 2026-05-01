import json
import uuid

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
        cleaned = normalize_whitespace_around_tags(cleaned)

        blocks = find_tool_call_blocks(cleaned)
        if not blocks:
            return ToolCallParseResult(None, content, [], cleaned)

        tool_calls = []
        parse_errors = []

        for i, (body, start, end) in enumerate(blocks):
            call, method = parse_tool_call_body(body)
            if not call or "name" not in call:
                parse_errors.append(f"tool_call[{i}] parse_failed")
                continue

            canonical_name = canonical_tool_name(call.get("name", ""), tools)
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                parsed_args = try_load_json(arguments)
                if isinstance(parsed_args, dict):
                    arguments = parsed_args

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
