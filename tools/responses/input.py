import uuid


def parse_responses_request(body: dict):
    from tools.responses.types import ResponsesRequest

    if not isinstance(body, dict):
        raise ValueError("Invalid JSON body")

    input_value = body.get("input")
    model = body.get("model")
    if input_value is None:
        raise ValueError("Missing 'input' field in request body")
    if not model:
        raise ValueError("Missing 'model' field in request body")

    tool_choice = body.get("tool_choice")
    if tool_choice == "auto":
        tool_choice = None
    elif tool_choice == "none":
        tool_choice = "none"
    elif tool_choice == "required":
        tool_choice = "required"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        if "function" not in tool_choice and tool_choice.get("name"):
            tool_choice = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }

    return ResponsesRequest(
        input=input_value,
        model=model,
        instructions=body.get("instructions"),
        previous_response_id=body.get("previous_response_id"),
        tools=body.get("tools") or [],
        tool_choice=tool_choice,
        stream=bool(body.get("stream", False)),
        max_output_tokens=body.get("max_output_tokens"),
        max_tokens=body.get("max_tokens"),
        store=body.get("store", True),
    )


def normalize_response_input(input_value, instructions=None):
    messages = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list):
        return messages

    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", "unknown"),
                "content": _coerce_content(item.get("output", "")),
            })
            continue

        if item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }],
            })
            continue

        role = item.get("role", "user")
        content = _coerce_content(item.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


def convert_responses_tools(tools):
    converted = []
    if not tools:
        return converted
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function")
        if isinstance(func, dict):
            converted.append(tool)
        else:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
    return converted


def _coerce_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("text", "input_text", "output_text")
        )
    return ""
