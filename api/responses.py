import json
import logging
import time
import uuid
from dataclasses import asdict

from flask import Blueprint, Response, jsonify, request, stream_with_context, current_app

from api.chat import _log_raw_request

logger = logging.getLogger(__name__)

from errors import openai_error
from model_config.registry import apply_model_mapping, parse_model_override, select_tool_adapter
from tools.adapters import get_adapter
from tools.responses.input import (
    convert_responses_tools,
    normalize_response_input,
    parse_responses_request,
)
from tools.responses.state import (
    get_response,
    load_history,
    mark_cancelled,
    store_history,
    store_response,
)
from tools.responses.types import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesResponse,
)
from provider.genai import (
    _extract_text_from_content,
    complete_usage,
    estimate_messages_token_count,
    estimate_token_count,
    stream_genai_response,
    stream_genai_response_with_tools,
)

responses_bp = Blueprint("responses", __name__)


def _serialize_response(response_obj: ResponsesResponse) -> dict:
    return {k: v for k, v in asdict(response_obj).items() if v is not None}


def _build_responses_usage(response_usage, *, messages, complete_content, complete_reasoning=""):
    usage = complete_usage(
        response_usage,
        prompt_tokens=estimate_messages_token_count(messages),
        completion_tokens=estimate_token_count(complete_content),
        reasoning_tokens=estimate_token_count(complete_reasoning) or None,
    )
    input_tokens = usage.get("prompt_tokens") or estimate_messages_token_count(messages)
    reasoning_tokens = estimate_token_count(complete_reasoning) or 0
    output_tokens = (usage.get("completion_tokens") or estimate_token_count(complete_content)) + reasoning_tokens
    usage["input_tokens"] = input_tokens
    usage["output_tokens"] = output_tokens
    usage["total_tokens"] = input_tokens + output_tokens
    return usage


def _is_error_chunk(data):
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not choices:
        return False
    delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
    content = delta.get("content")
    if isinstance(content, str) and content.startswith("[Error]"):
        return True
    return choices[0].get("finish_reason") == "error"


@responses_bp.route("/v1/responses", methods=["POST"])
def create_response():
    config = current_app.config["APP_CONFIG"]

    try:
        body = request.get_json() or {}
        if logger.isEnabledFor(logging.DEBUG):
            _log_raw_request(logger, f"req_{uuid.uuid4().hex[:16]}", "/v1/responses", body)
        req = parse_responses_request(body)
    except ValueError as exc:
        return openai_error(str(exc))

    prev_messages = load_history(req.previous_response_id)
    new_messages = normalize_response_input(req.input, req.instructions)
    messages = prev_messages + new_messages

    tools = convert_responses_tools(req.tools)
    has_tools = bool(tools)
    adapter = None

    if has_tools:
        clean_model, suffix_override = parse_model_override(req.model)
        clean_model = apply_model_mapping(clean_model, getattr(config, "model_mapping", {})) or clean_model
        header_override = (request.headers.get("X-Tool-Adapter") or "").strip().lower() or None
        adapter_override = header_override or suffix_override
        adapter_name = adapter_override or select_tool_adapter(clean_model, genai_record=None)
        adapter = get_adapter(adapter_name)
        messages = adapter.inject(messages, tools, req.tool_choice)
        req.model = clean_model

    chat_info = _extract_text_from_content(
        next((msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"), "")
    )

    if not chat_info:
        from provider.genai import _content_has_images
        has_any_image = any(
            _content_has_images(msg.get("content", ""))
            for msg in messages if msg.get("role") == "user"
        )
        if not has_any_image:
            return openai_error("No user message found in input")

    if req.stream:
        return _stream_response(req, messages, tools, adapter, chat_info, config)

    complete_content = ""
    complete_reasoning = ""
    response_usage = None
    max_tokens = req.max_output_tokens or req.max_tokens
    for line in stream_genai_response(chat_info, messages, req.model, max_tokens, config):
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                if isinstance(data.get("usage"), dict):
                    response_usage = data["usage"]
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = data.get("reasoning") or delta.get("reasoning_content", "")
                    if content:
                        complete_content += content
                    if reasoning:
                        complete_reasoning += reasoning
            except json.JSONDecodeError:
                pass

    tool_calls = None
    remaining_text = complete_content
    if has_tools and adapter:
        result = adapter.extract_tool_calls(complete_content, tools=tools)
        tool_calls = result.tool_calls
        remaining_text = result.remaining_text

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    output_items = []
    output_text = ""

    if tool_calls:
        for tc in tool_calls:
            output_items.append(ResponseFunctionToolCall(
                id=f"fc_{uuid.uuid4().hex[:12]}",
                call_id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", ""),
            ))
    else:
        text = remaining_text or complete_content
        output_text = text or ""
        output_items.append(ResponseOutputMessage(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            content=[ResponseOutputText(text=output_text or "")],
        ))

    response_obj = ResponsesResponse(
        id=response_id,
        created_at=created_at,
        previous_response_id=req.previous_response_id,
        model=req.model,
        output=output_items,
        output_text=output_text,
        usage=_build_responses_usage(
            response_usage,
            messages=messages,
            complete_content=complete_content,
            complete_reasoning=complete_reasoning,
        ),
    )

    if req.store:
        assistant_turn = _build_assistant_turn(tool_calls, remaining_text, complete_content)
        store_history(response_id, messages + [assistant_turn])
        store_response(response_id, _serialize_response(response_obj))

    return jsonify(_serialize_response(response_obj))


@responses_bp.route("/v1/responses/<response_id>", methods=["GET"])
def retrieve_response(response_id):
    payload = get_response(response_id)
    if payload is None:
        return openai_error(f"Response {response_id} not found", status=404)
    return jsonify(payload)


@responses_bp.route("/v1/responses/<response_id>/cancel", methods=["POST"])
def cancel_response(response_id):
    payload = mark_cancelled(response_id)
    if payload is None:
        return openai_error(f"Response {response_id} not found", status=404)
    return jsonify(payload)


def _build_assistant_turn(tool_calls, remaining_text, complete_content):
    if tool_calls:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    return {
        "role": "assistant",
        "content": remaining_text or complete_content,
    }


def _stream_response(req, messages, tools, adapter, chat_info, config):
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    def event(event_type, data):
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        response_stub = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": req.model,
            "output": [],
            "output_text": "",
            "usage": {
                "input_tokens": estimate_messages_token_count(messages),
                "output_tokens": 0,
                "total_tokens": estimate_messages_token_count(messages),
            },
        }
        yield event("response.created", {"type": "response.created", "response": response_stub})
        yield event("response.in_progress", {"type": "response.in_progress", "response": response_stub})

        complete_content = ""
        complete_reasoning = ""
        response_usage = None
        tool_calls = None
        remaining_text = ""
        upstream_error = None

        if tools and adapter:
            current_tool_calls = {}
            for line in stream_genai_response_with_tools(
                chat_info,
                messages,
                req.model,
                req.max_output_tokens or req.max_tokens,
                config,
                adapter=adapter,
                tools=tools,
            ):
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if _is_error_chunk(data):
                        upstream_error = data
                        break
                    if isinstance(data.get("usage"), dict):
                        response_usage = data["usage"]

                    if "choices" not in data or not data["choices"]:
                        continue

                    delta = data["choices"][0].get("delta", {})
                    if delta.get("content"):
                        complete_content += delta.get("content")
                    if delta.get("reasoning_content"):
                        complete_reasoning += delta.get("reasoning_content")
                    if delta.get("tool_calls"):
                        for item in delta.get("tool_calls", []):
                            call_id = item.get("id", "")
                            func = item.get("function", {})
                            current = current_tool_calls.get(call_id, {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if func.get("name"):
                                current["function"]["name"] = func.get("name")
                            if func.get("arguments"):
                                current["function"]["arguments"] = func.get("arguments")
                            current_tool_calls[call_id] = current
                if upstream_error:
                    break

            if current_tool_calls:
                tool_calls = list(current_tool_calls.values())

            remaining_text = complete_content
        else:
            max_tokens = req.max_output_tokens or req.max_tokens
            for line in stream_genai_response(chat_info, messages, req.model, max_tokens, config):
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if _is_error_chunk(data):
                        upstream_error = data
                        break
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

            remaining_text = complete_content

        output_items = []
        output_text = ""

        if tool_calls:
            for tc in tool_calls:
                output_items.append(ResponseFunctionToolCall(
                    id=f"fc_{uuid.uuid4().hex[:12]}",
                    call_id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                ))
        else:
            text = remaining_text or complete_content
            output_text = text or ""
            output_items.append(ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex[:12]}",
                content=[ResponseOutputText(text=output_text or "")],
            ))

        response_obj = ResponsesResponse(
            id=response_id,
            created_at=created_at,
            previous_response_id=req.previous_response_id,
            model=req.model,
            output=output_items,
            output_text=output_text,
            usage=_build_responses_usage(
                response_usage,
                messages=messages,
                complete_content=complete_content,
                complete_reasoning=complete_reasoning,
            ),
        )

        if req.store:
            assistant_turn = _build_assistant_turn(tool_calls, remaining_text, complete_content)
            store_history(response_id, messages + [assistant_turn])
            store_response(response_id, _serialize_response(response_obj))

        if upstream_error:
            err_text = ""
            choices = upstream_error.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                err_text = delta.get("content", "")
            serialized_response = _serialize_response(response_obj)
            serialized_response.setdefault("usage", {})
            serialized_response["usage"].setdefault("input_tokens", estimate_messages_token_count(messages))
            yield event("response.failed", {
                "type": "response.failed",
                "response_id": response_id,
                "error": {
                    "type": "upstream_error",
                    "message": err_text or "Upstream error",
                },
                "response": serialized_response,
            })
            yield "data: [DONE]\n\n"
            return

        for index, item in enumerate(output_items):
            item_in_progress = asdict(item)
            item_in_progress["status"] = "in_progress"
            yield event("response.output_item.added", {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": index,
                "item": item_in_progress,
            })

        if output_text:
            msg_item = output_items[0]
            yield event("response.content_part.added", {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                },
            })
            yield event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "delta": output_text,
            })
            yield event("response.output_text.done", {
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "text": output_text,
            })
            yield event("response.content_part.done", {
                "type": "response.content_part.done",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": output_text,
                    "annotations": [],
                },
            })
        elif tool_calls:
            for index, item in enumerate(output_items):
                yield event("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "item_id": item.id,
                    "output_index": index,
                    "name": item.name,
                    "call_id": item.call_id,
                    "arguments": item.arguments,
                })

        for index, item in enumerate(output_items):
            yield event("response.output_item.done", {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": index,
                "item": asdict(item),
            })

        yield event("response.completed", {"type": "response.completed", "response": _serialize_response(response_obj)})
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
