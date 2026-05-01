from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    public_id: str
    genai_id: str
    root_ai_type: str
    tool_adapter: str
    supports_reasoning: bool = False
    max_tokens: int = 30000
