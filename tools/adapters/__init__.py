from tools.adapters.base import ToolAdapter
from tools.adapters.generic import GenericAdapter
from tools.adapters.glm import GlmAdapter

_REGISTRY: dict[str, ToolAdapter] = {
    "generic": GenericAdapter(),
    "glm": GlmAdapter(),
}


def get_adapter(name: str) -> ToolAdapter:
    return _REGISTRY.get(name, _REGISTRY["generic"])
