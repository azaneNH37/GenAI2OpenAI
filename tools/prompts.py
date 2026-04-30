def inject_tool_prompt(messages, tools, tool_choice=None):
    from tools.adapters.generic import GenericAdapter

    return GenericAdapter().inject(messages, tools, tool_choice)
