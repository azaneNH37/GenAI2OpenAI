from dataclasses import dataclass, field


@dataclass
class ResponsesRequest:
    input: str | list[dict]
    model: str
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[dict] = field(default_factory=list)
    tool_choice: str | dict | None = None
    stream: bool = False
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    store: bool = True


@dataclass
class ResponseOutputText:
    type: str = "output_text"
    text: str = ""
    annotations: list = field(default_factory=list)


@dataclass
class ResponseOutputMessage:
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ResponseOutputText] = field(default_factory=list)
    status: str = "completed"


@dataclass
class ResponseFunctionToolCall:
    id: str
    call_id: str
    type: str = "function_call"
    name: str = ""
    arguments: str = ""
    status: str = "completed"


@dataclass
class ResponsesResponse:
    id: str
    created_at: int
    output: list
    usage: dict[str, int]
    object: str = "response"
    status: str = "completed"
    previous_response_id: str | None = None
    model: str = ""
    output_text: str = ""
