"""
上游 Token失效 自动 renew + 重试的单元测试 (离线)
用法: uv run tests/test_token_fallback_renew.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provider.genai import stream_genai_response


def _make_config(*, mode="credential", fallback_renew=True):
    cfg = MagicMock()
    cfg.token_manager.mode = mode
    cfg.token_manager.get_token.return_value = "token-v1"
    cfg.token_manager.force_refresh.return_value = "token-v2"
    cfg.fallback_renew = fallback_renew
    return cfg


def _expired_response():
    err_line = json.dumps({
        "success": False,
        "message": "Token失效，请重新登录",
        "code": 500,
    }).encode()
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter([err_line])
    return resp


def _ok_response(text="hello"):
    lines = [
        json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": None}]}).encode(),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
    ]
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


PATCHES = [
    patch("provider.genai.get_genai_id", return_value="test-ai"),
    patch("provider.genai.resolve_model", return_value=True),
    patch("provider.genai.get_root_ai_type", return_value="xinference"),
    patch("provider.genai.build_genai_headers", side_effect=lambda t: {"X-Access-Token": t}),
]


def with_patches(fn):
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in PATCHES:
            stack.enter_context(p)
        fn()


def test_credential_mode_renews_and_retries():
    """credential 模式 + 上游 Token失效 → 触发 force_refresh + 重试，最终成功。"""
    cfg = _make_config()
    with patch("requests.post", side_effect=[_expired_response(), _ok_response("hi")]) as mock_post:
        def run():
            chunks = list(stream_genai_response("q", [{"role": "user", "content": "q"}], "m", None, cfg))
            joined = "".join(chunks)
            assert "hi" in joined, f"应包含正文 hi: {joined}"
            assert "[DONE]" in joined
            assert "Token失效" not in joined, "不应把上游 token 错误暴露给客户端"
            assert mock_post.call_count == 2
            cfg.token_manager.force_refresh.assert_called_once()
            # 第二次请求应使用刷新后的 token
            second_headers = mock_post.call_args_list[1].kwargs["headers"]
            assert second_headers["X-Access-Token"] == "token-v2"
            print("[PASS] credential 模式 renew 后重试成功")
        with_patches(run)


def test_static_mode_does_not_renew():
    """static 模式：即便上游 Token失效 也不应自动 renew，直接报错给客户端。"""
    cfg = _make_config(mode="static")
    with patch("requests.post", side_effect=[_expired_response()]) as mock_post:
        def run():
            chunks = list(stream_genai_response("q", [{"role": "user", "content": "q"}], "m", None, cfg))
            joined = "".join(chunks)
            assert "Upstream error" in joined and "Token" in joined, "static 模式应把错误暴露给客户端"
            assert mock_post.call_count == 1
            cfg.token_manager.force_refresh.assert_not_called()
            print("[PASS] static 模式不自动 renew")
        with_patches(run)


def test_disabled_fallback_renew():
    """credential 模式但 fallback_renew=False → 不自动 renew。"""
    cfg = _make_config(fallback_renew=False)
    with patch("requests.post", side_effect=[_expired_response()]) as mock_post:
        def run():
            chunks = list(stream_genai_response("q", [{"role": "user", "content": "q"}], "m", None, cfg))
            joined = "".join(chunks)
            assert "Upstream error" in joined and "Token" in joined
            assert mock_post.call_count == 1
            cfg.token_manager.force_refresh.assert_not_called()
            print("[PASS] --disable-fallback-renew 生效")
        with_patches(run)


def test_only_retries_once():
    """如果 renew 后第二次仍然 Token失效，不再无限重试。"""
    cfg = _make_config()
    with patch("requests.post", side_effect=[_expired_response(), _expired_response()]) as mock_post:
        def run():
            chunks = list(stream_genai_response("q", [{"role": "user", "content": "q"}], "m", None, cfg))
            joined = "".join(chunks)
            assert "Upstream error" in joined, "二次失败应报错给客户端"
            assert mock_post.call_count == 2, "最多重试一次"
            assert cfg.token_manager.force_refresh.call_count == 1
            print("[PASS] 仅重试一次")
        with_patches(run)


def test_normal_response_unaffected():
    """正常响应路径：第一行就是有效 content，应正常工作。"""
    cfg = _make_config()
    with patch("requests.post", side_effect=[_ok_response("ok")]) as mock_post:
        def run():
            chunks = list(stream_genai_response("q", [{"role": "user", "content": "q"}], "m", None, cfg))
            joined = "".join(chunks)
            assert "ok" in joined
            assert "[DONE]" in joined
            assert mock_post.call_count == 1
            cfg.token_manager.force_refresh.assert_not_called()
            print("[PASS] 正常路径不受影响")
        with_patches(run)


if __name__ == "__main__":
    tests = [
        test_credential_mode_renews_and_retries,
        test_static_mode_does_not_renew,
        test_disabled_fallback_renew,
        test_only_retries_once,
        test_normal_response_unaffected,
    ]
    results = {}
    for t in tests:
        try:
            t()
            results[t.__name__] = True
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            results[t.__name__] = False

    passed = sum(results.values())
    total = len(results)
    print(f"\n  {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
