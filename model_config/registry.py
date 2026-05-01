from __future__ import annotations

from model_config.spec import ModelSpec

MODEL_SPECS: dict[str, ModelSpec] = {
    "glm-5.1": ModelSpec(
        public_id="glm-5.1",
        genai_id="chatglm",
        root_ai_type="xinference",
        tool_adapter="glm",
    ),
    "gpt-4.1": ModelSpec(
        public_id="gpt-4.1",
        genai_id="GPT-4.1",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-4.1-mini": ModelSpec(
        public_id="gpt-4.1-mini",
        genai_id="GPT-4.1-mini",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-o4-mini": ModelSpec(
        public_id="gpt-o4-mini",
        genai_id="o4-mini",
        root_ai_type="azure",
        tool_adapter="generic",
    ),
    "gpt-o3": ModelSpec(
        public_id="gpt-o3",
        genai_id="o3",
        root_ai_type="azure",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "deepseek-v4-flash": ModelSpec(
        public_id="deepseek-v4-flash",
        genai_id="deepseek-chat",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "deepseek-v4-pro": ModelSpec(
        public_id="deepseek-v4-pro",
        genai_id="deepseek-pro",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "qwen-instruct": ModelSpec(
        public_id="qwen-instruct",
        genai_id="qwen-instruct",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "minimax-m1": ModelSpec(
        public_id="minimax-m1",
        genai_id="MiniMax-M1",
        root_ai_type="xinference",
        tool_adapter="generic",
        supports_reasoning=True,
    ),
    "gpt-5.5": ModelSpec(
        public_id="gpt-5.5",
        genai_id="GPT-5.5",
        root_ai_type="azure",
        tool_adapter="generic",
        supports_reasoning=True,
    )
}

_ALIAS_MAP: dict[str, ModelSpec] = {}
for _spec in MODEL_SPECS.values():
    _ALIAS_MAP[_spec.public_id.lower()] = _spec
    _ALIAS_MAP[_spec.genai_id.lower()] = _spec

_EXTRA_ALIASES: dict[str, str] = {
    "glm": "glm-5.1",
    "chatglm": "glm-5.1",
    "gpt4.1": "gpt-4.1",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen-instruct",
}
for _alias, _target in _EXTRA_ALIASES.items():
    spec = MODEL_SPECS.get(_target)
    if spec:
        _ALIAS_MAP[_alias.lower()] = spec


def resolve_model(model_id: str | None) -> ModelSpec | None:
    return _ALIAS_MAP.get((model_id or "").lower())


def get_genai_id(model_id: str) -> str:
    spec = resolve_model(model_id)
    return spec.genai_id if spec else model_id


def get_root_ai_type(model_id: str, genai_record: dict | None = None) -> str:
    spec = resolve_model(model_id)
    if spec:
        return spec.root_ai_type
    if genai_record:
        return genai_record.get("rootAiType", "xinference")
    return "xinference"


def select_tool_adapter(model_id: str, genai_record: dict | None = None) -> str:
    spec = resolve_model(model_id)
    if spec:
        return spec.tool_adapter

    text = _build_model_text(model_id, genai_record)
    if "chatglm" in text or ("glm" in text and "xinference" in text):
        return "glm"
    return "generic"


def supports_reasoning(model_id: str) -> bool:
    spec = resolve_model(model_id)
    return spec.supports_reasoning if spec else False


def _build_model_text(model_id: str | None, record: dict | None) -> str:
    parts = [(model_id or "").lower()]
    if record:
        for key in ("aiType", "aiName", "rootModelName", "simpleName", "rootAiType"):
            value = record.get(key)
            if value:
                parts.append(str(value).lower())
    return " ".join(parts)
