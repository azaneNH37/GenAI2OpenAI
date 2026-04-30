"""
Responses API test script
Usage: uv run tests/test_responses.py --base-url http://localhost:5000 --model GPT-4.1
"""

import argparse
import json
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:5000")
parser.add_argument("--model", default="GPT-4.1")
args = parser.parse_args()

BASE_URL = args.base_url
MODEL = args.model

WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'Shanghai' or 'New York'",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
        },
        "required": ["location"],
    },
}


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _post(path, payload, stream=False):
    return requests.post(f"{BASE_URL}{path}", json=payload, stream=stream)


def _get(path):
    return requests.get(f"{BASE_URL}{path}")


def test_basic_text():
    print_separator("Test 1: Basic response (text)")
    resp = _post("/v1/responses", {
        "model": MODEL,
        "input": "Say hello in one short sentence.",
        "stream": False,
        "store": True,
    })
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    if data.get("object") != "response":
        print("[FAIL] Response object is not 'response'")
        return False, None

    output = data.get("output") or []
    output_text = data.get("output_text")
    if output and output[0].get("type") == "message" and output_text:
        print("[PASS] Basic text response ok")
        return True, data.get("id")

    print("[WARN] Unexpected output format")
    return False, data.get("id")


def test_get_response(response_id):
    print_separator("Test 2: Get response by id")
    if not response_id:
        print("[SKIP] No response id")
        return False

    resp = _get(f"/v1/responses/{response_id}")
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    if data.get("id") == response_id:
        print("[PASS] Retrieved response matches id")
        return True

    print("[FAIL] Response id mismatch")
    return False


def test_cancel_response(response_id):
    print_separator("Test 3: Cancel response")
    if not response_id:
        print("[SKIP] No response id")
        return False

    resp = _post(f"/v1/responses/{response_id}/cancel", {}, stream=False)
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    if data.get("status") == "cancelled":
        print("[PASS] Response cancelled")
        return True

    print("[FAIL] Cancel did not set status=cancelled")
    return False


def test_tool_call():
    print_separator("Test 4: Tool call (non-stream)")
    resp = _post("/v1/responses", {
        "model": MODEL,
        "input": "What's the weather in Beijing?",
        "tools": [WEATHER_TOOL_RESPONSES],
        "stream": False,
    })
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    output = data.get("output") or []
    if output and output[0].get("type") == "function_call":
        print("[PASS] Tool call detected")
        return True

    print("[WARN] No tool call in response output")
    return False


def test_previous_response_id():
    print_separator("Test 5: previous_response_id")
    resp1 = _post("/v1/responses", {
        "model": MODEL,
        "input": "Remember this word: pineapple.",
        "stream": False,
        "store": True,
    })
    data1 = resp1.json()
    response_id = data1.get("id")
    if not response_id:
        print("[FAIL] Missing response id")
        return False

    resp2 = _post("/v1/responses", {
        "model": MODEL,
        "input": "What word did I ask you to remember?",
        "previous_response_id": response_id,
        "stream": False,
    })
    data2 = resp2.json()
    print(json.dumps(data2, indent=2, ensure_ascii=False)[:2000])

    output_text = data2.get("output_text") or ""
    if output_text:
        print("[PASS] previous_response_id accepted")
        return True

    print("[WARN] previous_response_id response missing output_text")
    return False


def test_stream_text():
    print_separator("Test 6: Stream text")
    resp = _post("/v1/responses", {
        "model": MODEL,
        "input": "Answer in one sentence: What is the capital of France?",
        "stream": True,
    }, stream=True)

    events = []
    done = False
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:].strip()
        if data_str == "[DONE]":
            done = True
            break
        try:
            events.append(json.loads(data_str))
        except json.JSONDecodeError:
            pass

    types = [e.get("type") for e in events]
    print(f"  events: {types[:6]}")
    if done and "response.created" in types and "response.completed" in types:
        print("[PASS] Stream text events ok")
        return True

    print("[FAIL] Stream text missing expected events")
    return False


def test_stream_tool_call():
    print_separator("Test 7: Stream tool call")
    resp = _post("/v1/responses", {
        "model": MODEL,
        "input": "What's the weather in Tokyo?",
        "tools": [WEATHER_TOOL_RESPONSES],
        "stream": True,
    }, stream=True)

    events = []
    done = False
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:].strip()
        if data_str == "[DONE]":
            done = True
            break
        try:
            events.append(json.loads(data_str))
        except json.JSONDecodeError:
            pass

    types = [e.get("type") for e in events]
    print(f"  events: {types[:8]}")
    if done and "response.output_item.added" in types and "response.completed" in types:
        print("[PASS] Stream tool call events ok")
        return True

    print("[WARN] Stream tool call missing expected events")
    return False


if __name__ == "__main__":
    print(f"Testing against: {BASE_URL}")
    print(f"Model: {MODEL}")

    results = {}
    ok, response_id = test_basic_text()
    results["basic_text"] = ok
    results["get_response"] = test_get_response(response_id)
    results["cancel_response"] = test_cancel_response(response_id)
    results["tool_call"] = test_tool_call()
    results["previous_response_id"] = test_previous_response_id()
    results["stream_text"] = test_stream_text()
    results["stream_tool_call"] = test_stream_tool_call()

    print_separator("Summary")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
