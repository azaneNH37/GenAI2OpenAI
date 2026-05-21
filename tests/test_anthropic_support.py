import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from config import Config
from compat.anthropic import convert_anthropic_to_openai, map_anthropic_model_alias
from model_config.registry import resolve_model


class _TokenManager:
    def get_token(self):
        return "test-token"

    def force_refresh(self):
        return "refreshed-token"


def _app():
    return create_app(Config(
        token_manager=_TokenManager(),
        port=5000,
        api_key="secret",
        debug=False,
    ))


def _fake_stream(*_args, **_kwargs):
    yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n'
    yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n'
    yield "data: [DONE]\n\n"


class AnthropicSupportTests(unittest.TestCase):
    def test_model_alias_maps_claude_keywords(self):
        config = SimpleNamespace(
            claude_haiku_model="qwen-instruct",
            claude_sonnet_model="gpt-4.1",
            claude_opus_model="gpt-5.5",
        )
        self.assertEqual(map_anthropic_model_alias("claude-3-haiku", config), "qwen-instruct")
        self.assertEqual(map_anthropic_model_alias("claude-3-sonnet", config), "gpt-4.1")
        self.assertEqual(map_anthropic_model_alias("claude-3-opus", config), "gpt-5.5")

    def test_convert_anthropic_to_openai(self):
        config = SimpleNamespace(
            claude_haiku_model="qwen-instruct",
            claude_sonnet_model="gpt-4.1",
            claude_opus_model="gpt-5.5",
        )
        request = convert_anthropic_to_openai(
            {
                "model": "claude-3-7-sonnet-latest",
                "max_tokens": 128,
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "name": "lookup",
                    "description": "Lookup",
                    "input_schema": {"type": "object"},
                }],
                "tool_choice": {"type": "any"},
            },
            config,
        )

        self.assertEqual(request["model"], "gpt-4.1")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["tool_choice"], "required")
        self.assertEqual(request["tools"][0]["function"]["name"], "lookup")

    def test_codex_alias_is_resolved(self):
        self.assertIsNotNone(resolve_model("gpt-5-codex"))
        self.assertIsNotNone(resolve_model("codex"))

    def test_messages_endpoint_accepts_x_api_key(self):
        with patch("api.anthropic.stream_genai_response", side_effect=_fake_stream), patch(
            "api.anthropic.stream_genai_response_with_tools",
            side_effect=_fake_stream,
        ):
            resp = _app().test_client().post(
                "/v1/messages",
                headers={"x-api-key": "secret"},
                json={
                    "model": "claude-3-7-sonnet-latest",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["usage"]["input_tokens"], 7)
        self.assertEqual(payload["usage"]["output_tokens"], 2)

    def test_count_tokens_works(self):
        resp = _app().test_client().post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": "secret"},
            json={
                "model": "claude-3-7-sonnet-latest",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertIn("input_tokens", payload)


if __name__ == "__main__":
    unittest.main()
