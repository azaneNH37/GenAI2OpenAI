from tools.adapters.base import ToolAdapter
from tools.adapters.deepseek import DeepSeekAdapter, VARIANT_LEGACY, VARIANT_V4
from tools.adapters.generic import GenericAdapter
from tools.adapters.glm import GlmAdapter
from tools.adapters.minimax import MinimaxAdapter

_REGISTRY: dict[str, ToolAdapter] = {
    "generic": GenericAdapter(),
    "glm": GlmAdapter(),
    "minimax": MinimaxAdapter(),
    "deepseek_v4": DeepSeekAdapter(VARIANT_V4),
    "deepseek_legacy": DeepSeekAdapter(VARIANT_LEGACY),
    # 别名：deepseek -> v4（最新默认）
    "deepseek": DeepSeekAdapter(VARIANT_V4),
}


def get_adapter(name: str) -> ToolAdapter:
    return _REGISTRY.get(name, _REGISTRY["generic"])


def list_adapters() -> list[str]:
    """benchmarks 用：返回稳定的 adapter 名称列表。"""
    return ["generic", "glm", "minimax", "deepseek_v4", "deepseek_legacy"]
