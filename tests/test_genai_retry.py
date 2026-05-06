"""
DNS 重试逻辑单元测试 (离线，无需运行服务)
用法: uv run tests/test_genai_retry.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provider.genai import stream_genai_response


# --- helpers ---

def _make_config():
    cfg = MagicMock()
    cfg.token_manager.get_token.return_value = "fake-token"
    return cfg


def _make_response(content_chunks=None):
    """Mock streaming response that emits SSE lines."""
    lines = []
    for text in (content_chunks or []):
        chunk = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
        lines.append(json.dumps(chunk).encode())
    finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    lines.append(json.dumps(finish).encode())

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    return mock_resp


def _drain(gen):
    return list(gen)


PATCHES = [
    patch("provider.genai.get_genai_id", return_value="test-ai-type"),
    patch("provider.genai.resolve_model", return_value=True),
    patch("provider.genai.get_root_ai_type", return_value="xinference"),
    patch("provider.genai.build_genai_headers", return_value={}),
]


def with_patches(fn):
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in PATCHES:
            stack.enter_context(p)
        fn()


# --- tests ---

def test_normal_path():
    """正常路径：第一次就成功，无重试。"""
    mock_resp = _make_response(["hello"])
    cfg = _make_config()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        def run():
            chunks = _drain(stream_genai_response("hi", [{"role": "user", "content": "hi"}], "m", None, cfg))
            assert any("[DONE]" in c for c in chunks), "应有 [DONE]"
            assert mock_post.call_count == 1, f"期望调用 1 次，实际 {mock_post.call_count} 次"
            print("[PASS] 正常路径")
        with_patches(run)


def test_retry_once_then_succeed():
    """第 1 次 DNS 失败，第 2 次成功 → 最终返回正常数据。"""
    mock_resp = _make_response(["hello"])
    cfg = _make_config()
    dns_err = requests.exceptions.ConnectionError("DNS failure")

    with patch("requests.post", side_effect=[dns_err, mock_resp]) as mock_post, \
         patch("time.sleep") as mock_sleep:
        def run():
            chunks = _drain(stream_genai_response("hi", [{"role": "user", "content": "hi"}], "m", None, cfg))
            assert any("[DONE]" in c for c in chunks), "应有 [DONE]"
            assert mock_post.call_count == 2, f"期望调用 2 次，实际 {mock_post.call_count} 次"
            mock_sleep.assert_called_once_with(1)  # 第 1 次失败后等待 2^0 = 1s
            print("[PASS] 重试一次后成功")
        with_patches(run)


def test_retry_twice_then_succeed():
    """前 2 次 DNS 失败，第 3 次成功。"""
    mock_resp = _make_response(["hello"])
    cfg = _make_config()
    dns_err = requests.exceptions.ConnectionError("DNS failure")

    with patch("requests.post", side_effect=[dns_err, dns_err, mock_resp]) as mock_post, \
         patch("time.sleep") as mock_sleep:
        def run():
            chunks = _drain(stream_genai_response("hi", [{"role": "user", "content": "hi"}], "m", None, cfg))
            assert any("[DONE]" in c for c in chunks), "应有 [DONE]"
            assert mock_post.call_count == 3, f"期望调用 3 次，实际 {mock_post.call_count} 次"
            assert mock_sleep.call_args_list == [call(1), call(2)], "等待应为 1s, 2s"
            print("[PASS] 重试两次后成功")
        with_patches(run)


def test_all_attempts_fail_yields_error_chunk():
    """3 次全部 DNS 失败 → 不抛异常，yield error chunk 给客户端。"""
    cfg = _make_config()
    dns_err = requests.exceptions.ConnectionError("DNS failure")

    with patch("requests.post", side_effect=dns_err), \
         patch("time.sleep"):
        def run():
            chunks = _drain(stream_genai_response("hi", [{"role": "user", "content": "hi"}], "m", None, cfg))
            assert chunks, "应至少有一个 chunk"
            joined = " ".join(chunks)
            assert "error" in joined.lower() or "DNS" in joined, \
                f"期望错误信息，实际: {chunks}"
            print("[PASS] 3 次全部失败 → error chunk")
        with_patches(run)


def test_non_connection_error_not_retried():
    """非连接错误（如 Timeout）不应触发重试逻辑，直接进入 except Exception。"""
    cfg = _make_config()
    timeout_err = requests.exceptions.Timeout("timeout")

    with patch("requests.post", side_effect=timeout_err) as mock_post, \
         patch("time.sleep") as mock_sleep:
        def run():
            chunks = _drain(stream_genai_response("hi", [{"role": "user", "content": "hi"}], "m", None, cfg))
            assert chunks, "应至少有一个 chunk"
            assert mock_post.call_count == 1, "Timeout 不应重试"
            mock_sleep.assert_not_called()
            print("[PASS] Timeout 不重试")
        with_patches(run)


# --- runner ---

if __name__ == "__main__":
    tests = [
        test_normal_path,
        test_retry_once_then_succeed,
        test_retry_twice_then_succeed,
        test_all_attempts_fail_yields_error_chunk,
        test_non_connection_error_not_retried,
    ]

    results = {}
    for t in tests:
        name = t.__name__
        try:
            t()
            results[name] = True
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results[name] = False

    print(f"\n{'='*50}")
    passed = sum(v for v in results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
