def inject_with_renderers(
    messages,
    tool_prompt,
    render_tool_call_message,
    render_tool_results,
):
    new_messages = []
    has_system = False
    index = 0

    while index < len(messages):
        msg = messages[index]
        role = msg.get("role")

        if role == "system":
            new_messages.append({
                "role": "system",
                "content": msg.get("content", "") + "\n\n" + tool_prompt,
            })
            has_system = True
            index += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            new_messages.append({
                "role": "assistant",
                "content": render_tool_call_message(msg),
            })
            index += 1
            continue

        if role == "tool":
            tool_messages = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_messages.append(messages[index])
                index += 1
            new_messages.append({
                "role": "user",
                "content": render_tool_results(tool_messages),
            })
            continue

        new_messages.append(msg)
        index += 1

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages
