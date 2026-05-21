import json
import logging
import time
import uuid

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from api.chat import _log_raw_request
from compat.anthropic import (
    anthropic_error,
    convert_anthropic_to_openai,
    convert_openai_to_anthropic_response,
    estimate_anthropic_request_tokens,
    stream_openai_to_anthropic,
)
from model_config.registry import apply_model_mapping, parse_model_override, select_tool_adapter
from provider.genai import (
    _content_has_images,
    _extract_text_from_content,
    complete_usage,
    estimate_messages_token_count,
    estimate_token_count,
    stream_genai_response,
    stream_genai_response_with_tools,
)
from tools.adapters import get_adapter

logger = logging.getLogger(__name__)

anthropic_bp = Blueprint("anthropic", __name__)


def _serialize_message(resp):
    return json.loads(json.dumps(resp, ensure_ascii=False))


@anthropic_bp.route("/v1/messages", methods=["POST"])
def create_message():
    config = current_app.config["APP_CONFIG"]
    request_id = f"claude_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()

    try:
        body = request.get_json() or {}
        if logger.isEnabledFor(logging.DEBUG):
            _log_raw_request(logger, request_id, "/v1/messages", body)
        openai_request = convert_anthropic_to_openai(body, config)
    except ValueError as exc:
        return anthropic_error(str(exc))

    model = openai_request["model"]
    model = apply_model_mapping(model, getattr(config, "model_mapping", {})) or model
    original_request = {**body, "model": body.get("model"), "_estimator_model": model}

    header_override = (request.headers.get("X-Tool-Adapter") or "").strip().lower() or None
    _, suffix_override = parse_model_override(model)
    adapter_name = header_override or suffix_override
    tools = openai_request.get("tools") or []
    adapter = None
    if tools:
        adapter_name = adapter_name or select_tool_adapter(model, genai_record=None)
        adapter = get_adapter(adapter_name)
        openai_request["messages"] = adapter.inject(
            openai_request["messages"],
            tools,
            openai_request.get("tool_choice"),
        )

    chat_info = _extract_text_from_content(
        next((msg.get("content", "") for msg in reversed(openai_request["messages"]) if msg.get("role") == "user"), "")
    )

    if not chat_info:
        has_any_image = any(
            _content_has_images(msg.get("content", ""))
            for msg in openai_request["messages"] if msg.get("role") == "user"
        )
        if not has_any_image:
            return anthropic_error("No user message found in messages")

    if openai_request.get("stream"):
        return _stream_message(openai_request, original_request, chat_info, tools, adapter, config, request_id, start_time)

    complete_content = ""
    complete_reasoning = ""
    response_usage = None
    tool_call_map = {}
    max_tokens = openai_request.get("max_tokens", 30000)
    stream_fn = stream_genai_response_with_tools if tools and adapter else stream_genai_response
    stream_kwargs = dict(
        chat_info=chat_info,
        messages=openai_request["messages"],
        model=model,
        max_tokens=max_tokens,
        config=config,
    )
    if tools and adapter:
        stream_kwargs["adapter"] = adapter
        stream_kwargs["tools"] = tools

    for line in stream_fn(**stream_kwargs):
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("usage"), dict):
            response_usage = data["usage"]
        if "choices" not in data or not data["choices"]:
            continue
        delta = data["choices"][0].get("delta", {})
        content = delta.get("content", "")
        reasoning = data.get("reasoning") or delta.get("reasoning_content", "")
        if content:
            complete_content += content
        if reasoning:
            complete_reasoning += reasoning
        for item in delta.get("tool_calls", []) or []:
            call_id = item.get("id", "")
            func = item.get("function", {})
            current = tool_call_map.get(call_id, {
                "id": call_id,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if func.get("name"):
                current["function"]["name"] = func.get("name")
            if func.get("arguments"):
                current["function"]["arguments"] = func.get("arguments")
            tool_call_map[call_id] = current

    tool_calls = list(tool_call_map.values()) if tool_call_map else None
    remaining_text = complete_content
    if tools and adapter and not tool_calls:
        result = adapter.extract_tool_calls(complete_content, tools=tools)
        tool_calls = result.tool_calls
        remaining_text = result.remaining_text

    response = convert_openai_to_anthropic_response(
        {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "choices": [{
                "message": {
                    "content": remaining_text or complete_content,
                    "tool_calls": tool_calls or [],
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": complete_usage(
                response_usage,
                prompt_tokens=estimate_messages_token_count(openai_request["messages"]),
                completion_tokens=estimate_token_count(complete_content),
                reasoning_tokens=estimate_token_count(complete_reasoning) or None,
            ),
        },
        original_request,
    )
    if tools and adapter and tool_calls:
        response["content"] = [
            *([{"type": "text", "text": remaining_text}] if remaining_text else []),
            *[
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                }
                for tc in tool_calls
            ],
        ] or response["content"]

    return jsonify(_serialize_message(response))


@anthropic_bp.route("/v1/messages/count_tokens", methods=["POST"])
def count_tokens():
    config = current_app.config["APP_CONFIG"]
    try:
        body = request.get_json() or {}
        if logger.isEnabledFor(logging.DEBUG):
            _log_raw_request(logger, f"req_{uuid.uuid4().hex[:16]}", "/v1/messages/count_tokens", body)
    except ValueError as exc:
        return anthropic_error(str(exc))
    try:
        return jsonify({"input_tokens": estimate_anthropic_request_tokens(body)})
    except Exception as exc:
        return anthropic_error(str(exc), "api_error", 500)


def _stream_message(openai_request, original_request, chat_info, tools, adapter, config, request_id, start_time):
    def logged_stream():
        try:
            stream_fn = stream_genai_response_with_tools if tools and adapter else stream_genai_response
            stream_kwargs = dict(
                chat_info=chat_info,
                messages=openai_request["messages"],
                model=openai_request["model"],
                max_tokens=openai_request.get("max_tokens", 30000),
                config=config,
            )
            if tools and adapter:
                stream_kwargs["adapter"] = adapter
                stream_kwargs["tools"] = tools
            yield from stream_openai_to_anthropic(
                stream_fn(**stream_kwargs),
                original_request,
                logger,
            )
        finally:
            elapsed = time.monotonic() - start_time
            logger.info("[%s] completed in %.2fs", request_id, elapsed)

    return Response(
        stream_with_context(logged_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
