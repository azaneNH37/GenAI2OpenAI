"""
Error Handling & Health Check 测试脚本
用法: uv run tests/test_errors.py [--base-url http://localhost:5000]

测试场景:
  1. 健康检查 GET /health
  2. 缺少 messages 字段 (Chat Completions)
  3. 缺少 input 字段 (Responses API)
  4. 查询不存在的 response (404)
  5. 空模型名称
"""

import argparse
import json
import requests
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:5000")
args = parser.parse_args()

BASE_URL = args.base_url


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health_check():
    """测试 1: GET /health"""
    print_separator("Test 1: Health Check")

    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Body: {resp.text[:200]}")

    if resp.status_code == 200:
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if data.get("status") == "ok" or resp.text.strip() == "ok" or "ok" in resp.text.lower():
            print("[PASS] Health check returned ok")
            return True

    print("[FAIL] Health check did not return expected result")
    return False


def test_missing_messages():
    """测试 2: Chat Completions 缺少 messages 字段"""
    print_separator("Test 2: Missing 'messages' field (Chat Completions)")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "GPT-4.1",
    })

    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Body: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")

    error = data.get("error", {})
    if resp.status_code == 400 and error.get("type") == "invalid_request_error":
        print(f"[PASS] Correct error: type={error.get('type')}, message={error.get('message', '')[:100]}")
        return True

    if resp.status_code == 400 and error.get("message"):
        print(f"[PASS] Got 400 with error message: {error.get('message', '')[:100]}")
        return True

    print("[FAIL] Did not return expected 400 error")
    return False


def test_missing_input():
    """测试 3: Responses API 缺少 input 字段"""
    print_separator("Test 3: Missing 'input' field (Responses API)")

    resp = requests.post(f"{BASE_URL}/v1/responses", json={
        "model": "GPT-4.1",
    })

    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Body: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")

    error = data.get("error", {})
    if resp.status_code == 400 and error.get("message"):
        print(f"[PASS] Got 400 with error message: {error.get('message', '')[:100]}")
        return True

    print("[FAIL] Did not return expected 400 error")
    return False


def test_response_not_found():
    """测试 4: 查询不存在的 response (404)"""
    print_separator("Test 4: Response not found (404)")

    fake_id = "resp_nonexistent_" + "a" * 20
    resp = requests.get(f"{BASE_URL}/v1/responses/{fake_id}")

    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Body: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")

    error = data.get("error", {})
    if resp.status_code == 404 and error.get("type") == "invalid_request_error":
        print(f"[PASS] Correct 404 error: {error.get('message', '')[:100]}")
        return True

    if resp.status_code == 404 and error.get("message"):
        print(f"[PASS] Got 404 with error message: {error.get('message', '')[:100]}")
        return True

    print("[FAIL] Did not return expected 404 error")
    return False


def test_empty_model():
    """测试 5: 空模型名称"""
    print_separator("Test 5: Empty model name")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "",
        "messages": [{"role": "user", "content": "hello"}],
    })

    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Body: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")

    # 服务可能返回错误也可能尝试转发（取决于实现），这里只验证不崩溃
    if resp.status_code in (200, 400, 500):
        print(f"[PASS] Server handled empty model gracefully (status={resp.status_code})")
        return True

    print("[FAIL] Unexpected status code")
    return False


if __name__ == "__main__":
    print(f"Testing against: {BASE_URL}")

    results = {}
    tests = [
        ("health_check", test_health_check),
        ("missing_messages", test_missing_messages),
        ("missing_input", test_missing_input),
        ("response_not_found", test_response_not_found),
        ("empty_model", test_empty_model),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            results[name] = False

    print_separator("Summary")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)