import json
import logging
import time
import uuid
from datetime import datetime

from flask import Blueprint, current_app, request, jsonify, stream_with_context, Response

from errors import openai_error
from model_config.registry import parse_model_override, select_tool_adapter
from tools.adapters import get_adapter
from provider.genai import (
    complete_usage,
    convert_messages_to_genai_format,
    estimate_messages_token_count,
    estimate_token_count,
    stream_genai_response,
    stream_genai_response_with_tools,
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


def _summarize_part(part, max_text=300, max_url=120):
    if not isinstance(part, dict):
        return {"_raw_type": type(part).__name__, "_repr": repr(part)[:max_text]}
    ptype = part.get("type")
    summary = {"type": ptype, "keys": sorted(part.keys())}
    if ptype in ("text", "input_text", "output_text"):
        text = part.get("text", "")
        summary["text_len"] = len(text) if isinstance(text, str) else None
        summary["text_preview"] = (text[:max_text] + "...") if isinstance(text, str) and len(text) > max_text else text
    elif ptype in ("image_url", "input_image"):
        img = part.get("image_url") or part.get("image") or {}
        url = img.get("url") if isinstance(img, dict) else (img if isinstance(img, str) else None)
        if url is None:
            url = part.get("url") or part.get("data") or part.get("source")
        if isinstance(url, str):
            summary["image_scheme"] = url.split(":", 1)[0] if ":" in url else "unknown"
            summary["image_url_len"] = len(url)
            summary["image_url_preview"] = url[:max_url]
        else:
            summary["image_raw"] = repr(url)[:max_text]
        if isinstance(img, dict) and "detail" in img:
            summary["detail"] = img["detail"]
    else:
        summary["raw_preview"] = repr(part)[:max_text]
    return summary


def _summarize_content(content):
    if isinstance(content, str):
        return {"kind": "str", "len": len(content), "preview": content[:300]}
    if isinstance(content, list):
        return {
            "kind": "list",
            "n_parts": len(content),
            "parts": [_summarize_part(p) for p in content],
        }
    return {"kind": type(content).__name__, "repr": repr(content)[:300]}


def _log_raw_request(log, request_id, endpoint, body):
    if not isinstance(body, dict):
        log.debug("[%s] %s raw body type=%s repr=%s",
                  request_id, endpoint, type(body).__name__, repr(body)[:500])
        return
    top_keys = sorted(body.keys())
    log.debug("[%s] %s top-level keys: %s", request_id, endpoint, top_keys)
    for k, v in body.items():
        if k == "messages" or k == "input":
            continue
        log.debug("[%s]   %s = %s", request_id, k, repr(v)[:300])
    messages = body.get("messages") or body.get("input") or []
    if not isinstance(messages, list):
        log.debug("[%s] messages is not a list: type=%s", request_id, type(messages).__name__)
        return
    log.debug("[%s] === %d message(s) ===", request_id, len(messages))
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            log.debug("[%s] msg[%d] not a dict: %s", request_id, i, repr(msg)[:300])
            continue
        role = msg.get("role")
        extra_keys = [k for k in msg.keys() if k not in ("role", "content")]
        content_summary = _summarize_content(msg.get("content", ""))
        log.debug("[%s] msg[%d] role=%s extra_keys=%s content=%s",
                  request_id, i, role, extra_keys, content_summary)


@chat_bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    config = current_app.config["APP_CONFIG"]
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()
    defer_completion_log = False

    def log_completion():
        elapsed = time.monotonic() - start_time
        logger.info("[%s] completed in %.2fs", request_id, elapsed)

    try:
        req_data = request.get_json()

        if logger.isEnabledFor(logging.DEBUG):
            _log_raw_request(logger, request_id, "/v1/chat/completions", req_data)

        if not req_data or 'messages' not in req_data:
            return openai_error("Missing 'messages' field in request body")

        messages = req_data.get('messages', [])
        raw_model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        tools = req_data.get('tools', None)
        tool_choice = req_data.get('tool_choice', None)

        # 支持 model@adapter 后缀 + X-Tool-Adapter header 强制覆盖适配器
        model, suffix_override = parse_model_override(raw_model)
        header_override = (request.headers.get("X-Tool-Adapter") or "").strip().lower() or None
        adapter_override = header_override or suffix_override

        has_tools = tools and len(tools) > 0

        logger.info("[%s] model=%s stream=%s tools=%s messages=%d adapter_override=%s",
                     request_id, model, stream, bool(has_tools), len(messages), adapter_override)

        adapter = None
        if has_tools:
            adapter_name = adapter_override or select_tool_adapter(model, genai_record=None)
            adapter = get_adapter(adapter_name)
            messages = adapter.inject(messages, tools, tool_choice)

        chat_info = convert_messages_to_genai_format(messages)

        if not chat_info:
            from provider.genai import _content_has_images
            has_any_image = any(
                _content_has_images(msg.get("content", ""))
                for msg in messages if msg.get("role") == "user"
            )
            if not has_any_image:
                return openai_error("No user message found in 'messages'")

        if stream:
            if has_tools:
                gen = stream_genai_response_with_tools(
                    chat_info,
                    messages,
                    model,
                    max_tokens,
                    config,
                    adapter=adapter,
                    tools=tools,
                )
            else:
                gen = stream_genai_response(
                    chat_info, messages, model, max_tokens, config
                )

            def logged_stream():
                try:
                    yield from gen
                finally:
                    log_completion()

            defer_completion_log = True
            return Response(
                stream_with_context(logged_stream()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        else:
            complete_content = ""
            complete_reasoning = ""
            response_usage = None
            for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str == '[DONE]':
                        continue
                    try:
                        data = json.loads(data_str)
                        if isinstance(data.get('usage'), dict):
                            response_usage = data['usage']
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            reasoning = data.get('reasoning') or delta.get('reasoning_content', '')
                            if content:
                                complete_content += content
                            if reasoning:
                                complete_reasoning += reasoning
                    except json.JSONDecodeError:
                        pass

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if has_tools and adapter:
                result = adapter.extract_tool_calls(complete_content, tools=tools)
                tool_calls = result.tool_calls
                remaining_text = result.remaining_text
                if result.parse_errors:
                    logger.warning("[%s] tool call parse errors: %s", request_id, result.parse_errors)
            else:
                tool_calls, remaining_text = None, complete_content

            if tool_calls:
                message_obj = {
                    "role": "assistant",
                    "content": remaining_text,
                    "tool_calls": tool_calls
                }
                if complete_reasoning:
                    message_obj["reasoning_content"] = complete_reasoning
                finish_reason = "tool_calls"
            else:
                message_obj = {
                    "role": "assistant",
                    "content": complete_content
                }
                if complete_reasoning:
                    message_obj["reasoning_content"] = complete_reasoning
                finish_reason = "stop"

            usage = complete_usage(
                response_usage,
                prompt_tokens=estimate_messages_token_count(messages),
                completion_tokens=estimate_token_count(complete_content),
                reasoning_tokens=estimate_token_count(complete_reasoning) or None,
            )

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason
                }],
                "usage": usage
            }
            return jsonify(response)

    except Exception as e:
        logger.exception("[%s] Unhandled error", request_id)
        return openai_error(
            str(e),
            error_type="server_error",
            code="internal_error",
            status=500
        )
    finally:
        if not defer_completion_log:
            log_completion()
