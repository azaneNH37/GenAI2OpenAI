import json
import time
import uuid
from dataclasses import asdict

from flask import Blueprint, Response, jsonify, request, stream_with_context, current_app

from errors import openai_error
from model_config.registry import select_tool_adapter
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
from provider.genai import stream_genai_response, stream_genai_response_with_tools

responses_bp = Blueprint("responses", __name__)


def _serialize_response(response_obj: ResponsesResponse) -> dict:
    return {k: v for k, v in asdict(response_obj).items() if v is not None}


@responses_bp.route("/v1/responses", methods=["POST"])
def create_response():
    config = current_app.config["APP_CONFIG"]

    try:
        body = request.get_json() or {}
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
        adapter_name = select_tool_adapter(req.model, genai_record=None)
        adapter = get_adapter(adapter_name)
        messages = adapter.inject(messages, tools, req.tool_choice)

    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break

    if not chat_info:
        return openai_error("No user message found in input")

    if req.stream:
        return _stream_response(req, messages, tools, adapter, chat_info, config)

    complete_content = ""
    complete_reasoning = ""
    for line in stream_genai_response(chat_info, messages, req.model, req.max_output_tokens or req.max_tokens, config):
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
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
        usage={
            "prompt_tokens": 0,
            "completion_tokens": len(complete_content),
            "total_tokens": len(complete_content),
        },
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

    def event(data):
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        response_stub = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": req.model,
            "output": [],
            "output_text": "",
        }
        yield event({"type": "response.created", "response": response_stub})

        complete_content = ""
        complete_reasoning = ""
        tool_calls = None
        remaining_text = ""

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

            if current_tool_calls:
                tool_calls = list(current_tool_calls.values())

            remaining_text = complete_content
        else:
            for line in stream_genai_response(chat_info, messages, req.model, req.max_output_tokens or req.max_tokens, config):
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

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
            usage={
                "prompt_tokens": 0,
                "completion_tokens": len(complete_content),
                "total_tokens": len(complete_content),
            },
        )

        if req.store:
            assistant_turn = _build_assistant_turn(tool_calls, remaining_text, complete_content)
            store_history(response_id, messages + [assistant_turn])
            store_response(response_id, _serialize_response(response_obj))

        for index, item in enumerate(output_items):
            yield event({
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": index,
                "item": asdict(item),
            })

        if output_text:
            msg_item = output_items[0]
            yield event({
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "delta": output_text,
            })
            yield event({
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": msg_item.id,
                "output_index": 0,
                "content_index": 0,
                "text": output_text,
            })
        elif tool_calls:
            for index, item in enumerate(output_items):
                yield event({
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "item_id": item.id,
                    "output_index": index,
                    "name": item.name,
                    "call_id": item.call_id,
                    "arguments": item.arguments,
                })

        for index, item in enumerate(output_items):
            yield event({
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": index,
                "item": asdict(item),
            })

        yield event({"type": "response.completed", "response": _serialize_response(response_obj)})
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
