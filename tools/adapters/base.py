class ToolAdapter:
    """
    per-model tool call adapter, stateless, shared instance.
    """

    open_tags: tuple[str, ...] = ("<tool_call>",)

    def inject(self, messages, tools, tool_choice=None):
        raise NotImplementedError

    def extract_tool_calls(self, content, tools=None):
        raise NotImplementedError
