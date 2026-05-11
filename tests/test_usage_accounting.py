import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import create_app
from config import Config
from provider import genai


class _TokenManager:
    def get_token(self):
        return "test-token"

    def force_refresh(self):
        return "refreshed-token"


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class _BlockingLines:
    def __init__(self):
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.closed.wait(timeout=1.0)
        raise StopIteration

    def close(self):
        self.closed.set()


class _FakeToolAdapter:
    open_tags = ["<tool_call>"]

    def extract_tool_calls(self, _raw, tools=None):
        return SimpleNamespace(
            tool_calls=[{
                "id": "call_weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
            }],
            remaining_text="",
            parse_errors=[],
        )


def _app():
    return create_app(Config(
        token_manager=_TokenManager(),
        port=5000,
        api_key=None,
        debug=False,
    ))


def _sse_json_events(raw_text):
    events = []
    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


class UsageAccountingTests(unittest.TestCase):
    def test_extract_usage_normalizes_genai_camel_case_fields(self):
        usage = genai.extract_usage_from_genai({
            "other": json.dumps({
                "inputTokens": "12",
                "outputTokens": 34,
                "reasoningTokens": 5,
                "cacheTokens": 6,
                "totalTokens": 51,
            })
        })

        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 34)
        self.assertEqual(usage["total_tokens"], 51)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 5)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 6)

    def test_stream_sends_prompt_token_estimate_and_preserves_post_finish_usage(self):
        calls = []

        def fake_post(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse([
                b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                b'data: {"other":"{\\"promptTokens\\":11}"}',
            ])

        config = SimpleNamespace(token_manager=_TokenManager())
        messages = [{"role": "user", "content": "你好，请介绍 GenAI"}]

        with patch.object(genai.requests, "post", side_effect=fake_post):
            chunks = list(genai.stream_genai_response(
                chat_info="你好，请介绍 GenAI",
                messages=messages,
                model="deepseek-v4-pro",
                max_tokens=None,
                config=config,
            ))

        self.assertGreater(calls[0][1]["json"]["promptTokens"], 0)
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        final_payload = json.loads(chunks[-2].removeprefix("data: "))
        self.assertEqual(final_payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(final_payload["usage"]["prompt_tokens"], 11)
        self.assertGreater(final_payload["usage"]["completion_tokens"], 0)
        self.assertEqual(
            final_payload["usage"]["total_tokens"],
            final_payload["usage"]["prompt_tokens"] + final_payload["usage"]["completion_tokens"],
        )

    def test_post_finish_drain_times_out_and_closes_response(self):
        blocking_lines = _BlockingLines()

        drained = list(genai._drain_post_finish_lines(
            blocking_lines,
            timeout=0.01,
            close=blocking_lines.close,
        ))

        self.assertEqual(drained, [])
        self.assertTrue(blocking_lines.closed.is_set())

    def test_tool_stream_terminal_chunk_preserves_usage(self):
        def fake_stream(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"<tool_call>{}</tool_call>"},"finish_reason":null}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}\n\n'
            )
            yield "data: [DONE]\n\n"

        with patch.object(genai, "stream_genai_response", side_effect=fake_stream):
            chunks = list(genai.stream_genai_response_with_tools(
                chat_info="weather",
                messages=[{"role": "user", "content": "weather"}],
                model="deepseek-v4-pro",
                max_tokens=None,
                config=SimpleNamespace(token_manager=_TokenManager()),
                adapter=_FakeToolAdapter(),
                tools=[],
            ))

        final_payload = json.loads(chunks[-2].removeprefix("data: "))
        self.assertEqual(final_payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(final_payload["usage"], {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        })

    def test_non_stream_chat_response_uses_stream_usage(self):
        def fake_stream(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}\n\n'
            )
            yield "data: [DONE]\n\n"

        with patch("api.chat.stream_genai_response", side_effect=fake_stream):
            response = _app().test_client().post("/v1/chat/completions", json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["usage"], {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13,
        })

    def test_non_stream_responses_response_uses_stream_usage(self):
        def fake_stream(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n'
            )
            yield "data: [DONE]\n\n"

        with patch("api.responses.stream_genai_response", side_effect=fake_stream):
            response = _app().test_client().post("/v1/responses", json={
                "model": "deepseek-v4-pro",
                "input": "hello",
                "stream": False,
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["output_text"], "hello")
        self.assertEqual(payload["usage"], {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
        })

    def test_stream_responses_completed_event_uses_stream_usage(self):
        def fake_stream(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}\n\n'
            )
            yield "data: [DONE]\n\n"

        with patch("api.responses.stream_genai_response", side_effect=fake_stream):
            response = _app().test_client().post("/v1/responses", json={
                "model": "deepseek-v4-pro",
                "input": "hello",
                "stream": True,
            }, buffered=True)

        self.assertEqual(response.status_code, 200)
        events = _sse_json_events(response.get_data(as_text=True))
        completed = next(event for event in events if event.get("type") == "response.completed")
        self.assertEqual(completed["response"]["usage"], {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8,
        })


if __name__ == "__main__":
    unittest.main()
