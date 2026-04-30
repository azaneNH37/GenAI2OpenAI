"""
Tool Calling 测试脚本
用法: uv run tests/test_tool_calling.py [--base-url http://localhost:5000] [--model GPT-4.1]

测试场景:
  0. 模型列表
  1. 单次 tool call (天气查询)
  2. 多次 tool call (并行查天气)
  3. 多轮对话 (tool call -> tool result -> 最终回答)
  4. 无 tool 调用 (普通问题不应触发 tool)
  5. 流式 tool call
  6. 流式 + tools 但无需调用 (验证真流式)
  7. Tag 前缀检测 (离线单元测试)
  8. 本地文件读取 tool call
  9. tool_choice=none
  10. tool_choice=required
  11. tool_choice=specific (指定函数名)
  12. 复杂参数类型 (integer/boolean/array)
"""

import argparse
import json
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://localhost:5000')
parser.add_argument('--model', default='GPT-4.1')
args = parser.parse_args()

BASE_URL = args.base_url
MODEL = args.model

# 定义测试用的 tools
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Shanghai' or 'New York'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a local file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file"
                }
            },
            "required": ["path"]
        }
    }
}

SEARCH_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search academic papers on a given topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results"
                },
                "include_abstracts": {
                    "type": "boolean",
                    "description": "Whether to include paper abstracts"
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to return"
                }
            },
            "required": ["query"]
        }
    }
}


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_list_models():
    """测试 0: 获取模型列表"""
    print_separator("Test 0: List Models")

    resp = requests.get(f"{BASE_URL}/v1/models")
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if data.get("object") != "list":
        print("\n[FAIL] Response object is not 'list'")
        return False

    models = data.get("data", [])
    if not models:
        print("\n[FAIL] No models returned")
        return False

    model_ids = [m["id"] for m in models]
    print(f"\n[PASS] {len(models)} models available: {model_ids}")
    return True


def test_single_tool_call():
    """测试 1: 单次 tool call"""
    print_separator("Test 1: Single Tool Call (Weather)")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather in Shanghai?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    if msg.get("tool_calls"):
        print("\n[PASS] Tool calls detected:")
        for tc in msg["tool_calls"]:
            print(f"  - {tc['function']['name']}({tc['function']['arguments']})")
        return True
    else:
        print("\n[FAIL] No tool calls in response")
        print(f"  Content: {msg.get('content', '')[:200]}")
        return False


def test_multiple_tools():
    """测试 2: 提供多个 tools，看模型是否选对"""
    print_separator("Test 2: Multiple Tools Available")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is 123 * 456 + 789?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    if msg.get("tool_calls"):
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"\n[INFO] Tools called: {names}")
        if "calculate" in names:
            print("[PASS] Correctly chose 'calculate' tool")
            return True
        else:
            print("[WARN] Did not choose 'calculate' - chose different tool")
            return False
    else:
        print("\n[FAIL] No tool calls in response")
        return False


def test_multi_turn():
    """测试 3: 多轮 tool calling (call -> result -> final answer)"""
    print_separator("Test 3: Multi-turn Tool Calling")

    # 第一轮：用户提问
    print("--- Round 1: User asks about weather ---")
    resp1 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather like in Beijing right now?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data1 = resp1.json()
    choice1 = data1.get("choices", [{}])[0]
    msg1 = choice1.get("message", {})

    if not msg1.get("tool_calls"):
        print("[FAIL] Round 1 did not produce tool calls")
        print(json.dumps(data1, indent=2, ensure_ascii=False))
        return False

    tc = msg1["tool_calls"][0]
    print(f"  Tool called: {tc['function']['name']}({tc['function']['arguments']})")

    # 第二轮：把 tool 结果传回去
    print("\n--- Round 2: Sending tool result back ---")
    resp2 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather like in Beijing right now?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": msg1["tool_calls"]
            },
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({
                    "location": "Beijing",
                    "temperature": 22,
                    "unit": "celsius",
                    "condition": "Partly cloudy",
                    "humidity": 45
                })
            }
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data2 = resp2.json()
    print(json.dumps(data2, indent=2, ensure_ascii=False))

    choice2 = data2.get("choices", [{}])[0]
    msg2 = choice2.get("message", {})
    if msg2.get("content") and not msg2.get("tool_calls"):
        print(f"\n[PASS] Final answer: {msg2['content'][:200]}")
        return True
    elif msg2.get("tool_calls"):
        print("\n[WARN] Model called another tool instead of answering")
        return False
    else:
        print("\n[FAIL] Empty response")
        return False


def test_no_tool_needed():
    """测试 4: 普通问题不应触发 tool call"""
    print_separator("Test 4: No Tool Needed")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
        "stream": False
    })

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    if not msg.get("tool_calls"):
        print(f"[PASS] No tool called. Answer: {msg.get('content', '')[:200]}")
        return True
    else:
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"[WARN] Unexpected tool call: {names}")
        return False


def test_stream_tool_call():
    """测试 5: 流式 tool calling"""
    print_separator("Test 5: Streaming Tool Call")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": True
    }, stream=True)

    full_content = ""
    tool_calls_found = []

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line_str.startswith('data: '):
            continue
        data_str = line_str[6:].strip()
        if data_str == '[DONE]':
            print("  [DONE]")
            break

        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish = chunk.get("choices", [{}])[0].get("finish_reason")

            if delta.get("content"):
                full_content += delta["content"]
                print(f"  content: {delta['content'][:80]}")

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tool_calls_found.append(tc)
                    print(f"  tool_call: {tc['function']['name']}({tc['function']['arguments']})")

            if finish:
                print(f"  finish_reason: {finish}")

        except json.JSONDecodeError:
            pass

    if tool_calls_found:
        print(f"\n[PASS] Stream tool calls detected: {len(tool_calls_found)}")
        return True
    elif full_content and '<tool_call>' in full_content:
        print(f"\n[WARN] Tool call in raw text but not parsed")
        return False
    else:
        print(f"\n[FAIL] No tool calls in stream")
        print(f"  Content: {full_content[:200]}")
        return False


def test_stream_no_tool_needed():
    """测试 6: 流式 tool calling — 普通问题应真正流式输出（不缓冲）"""
    print_separator("Test 6: Streaming With Tools (No Tool Needed)")

    import time
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ],
        "tools": [WEATHER_TOOL],
        "stream": True
    }, stream=True)

    chunks = []
    first_chunk_time = None
    start_time = time.monotonic()

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line_str.startswith('data: '):
            continue
        data_str = line_str[6:].strip()
        if data_str == '[DONE]':
            break

        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish = chunk.get("choices", [{}])[0].get("finish_reason")

            if delta.get("content"):
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                chunks.append(delta["content"])

        except json.JSONDecodeError:
            pass

    total_time = time.monotonic() - start_time
    ttfb = (first_chunk_time - start_time) if first_chunk_time else total_time
    full_content = "".join(chunks)

    print(f"  Chunks: {len(chunks)}")
    print(f"  TTFB: {ttfb:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"  Content: {full_content[:200]}")

    if len(chunks) > 1:
        print(f"\n[PASS] True streaming: {len(chunks)} chunks (not buffered into 1)")
        return True
    elif len(chunks) == 1 and full_content:
        print(f"\n[WARN] Only 1 chunk — may be buffered")
        return False
    else:
        print(f"\n[FAIL] No content received")
        return False


def test_local_file_tool_call():
    """测试 8: 本地文件读取 tool call"""
    print_separator("Test 8: Local File Tool Call")

    target_path = Path(__file__).resolve()
    prompt = (
        "Read the file at the given path and return only the first line. "
        f"Path: {target_path}"
    )

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "tools": [READ_FILE_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        print("\n[FAIL] No tool calls in response")
        return False

    tc = tool_calls[0]
    args_raw = tc.get("function", {}).get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        args = {}

    path_arg = args.get("path", "")
    print(f"  tool_call: {tc['function']['name']}({args_raw})")
    if tc.get("function", {}).get("name") != "read_file":
        print("\n[FAIL] Tool name mismatch")
        return False

    if not path_arg:
        print("\n[FAIL] Missing path argument")
        return False

    if "\\" in path_arg and "\\\\" in path_arg:
        print("\n[WARN] Detected double-escaped backslashes in path")

    if not Path(path_arg).exists():
        print(f"\n[FAIL] Path does not exist: {path_arg}")
        return False

    print("\n[PASS] Tool call includes valid existing path")
    return True


def test_tool_choice_none():
    """测试 9: tool_choice=none — 即使提供了工具也不应调用"""
    print_separator("Test 9: tool_choice=none")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
        "tool_choice": "none",
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    if not msg.get("tool_calls") and msg.get("content"):
        print(f"[PASS] No tool called with tool_choice=none. Answer: {msg.get('content', '')[:200]}")
        return True
    elif msg.get("tool_calls"):
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"[FAIL] tool_choice=none but tools were called: {names}")
        return False
    else:
        print("[FAIL] No tool calls but also no content")
        return False


def test_tool_choice_required():
    """测试 10: tool_choice=required — 必须调用至少一个工具"""
    print_separator("Test 10: tool_choice=required")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Shanghai?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
        "tool_choice": "required",
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    if msg.get("tool_calls") and len(msg["tool_calls"]) >= 1:
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"[PASS] tool_choice=required triggered tool call(s): {names}")
        return True
    else:
        print("[WARN] tool_choice=required but no tool calls (model may not follow prompt injection)")
        return False


def test_tool_choice_specific():
    """测试 11: tool_choice=specific — 指定调用某个函数"""
    print_separator("Test 11: tool_choice=specific (web_search)")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Find information about the Olympic Games"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []

    if tool_calls:
        names = [tc['function']['name'] for tc in tool_calls]
        if "web_search" in names:
            print(f"[PASS] tool_choice=specific correctly called web_search: {names}")
            return True
        else:
            print(f"[WARN] tool_choice=specific for web_search but called: {names} (prompt injection not strongly enforced)")
            return False
    else:
        print("[WARN] tool_choice=specific but no tool calls (prompt injection not strongly enforced)")
        return False


def test_complex_parameter_types():
    """测试 12: 复杂参数类型 (integer/boolean/array)"""
    print_separator("Test 12: Complex Parameter Types (int/bool/array)")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Search for recent AI papers, limit 5 results, include abstracts"}
        ],
        "tools": [SEARCH_PAPERS_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []

    if not tool_calls:
        print("[FAIL] No tool calls in response")
        return False

    tc = tool_calls[0]
    args_raw = tc.get("function", {}).get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        print(f"[FAIL] Arguments JSON parse error: {args_raw[:200]}")
        return False

    print(f"  tool_call: {tc['function']['name']}({args_raw})")
    print(f"  parsed args: {args}")

    has_query = "query" in args
    if not has_query:
        print("[FAIL] Missing required 'query' argument")
        return False

    # Check integer type coercion for 'limit'
    limit_val = args.get("limit")
    if limit_val is not None:
        if isinstance(limit_val, int) and not isinstance(limit_val, bool):
            print(f"  [OK] limit is integer: {limit_val}")
        else:
            print(f"  [WARN] limit is not integer: {type(limit_val).__name__} = {limit_val}")

    # Check boolean type for 'include_abstracts'
    abs_val = args.get("include_abstracts")
    if abs_val is not None:
        if isinstance(abs_val, bool):
            print(f"  [OK] include_abstracts is boolean: {abs_val}")
        else:
            print(f"  [WARN] include_abstracts is not boolean: {type(abs_val).__name__} = {abs_val}")

    print(f"[PASS] Complex parameter types tool call parsed successfully")
    return True


def test_tag_prefix_detection():
    """测试 7: _tag_prefix_len 单元测试（离线，不需要服务器）"""
    print_separator("Test 7: Tag Prefix Detection (Unit Test)")

    # 可从 tools.parsing 导入，此处内联以便独立运行
    def _tag_prefix_len(text, tag):
        max_len = min(len(tag) - 1, len(text))
        for length in range(max_len, 0, -1):
            if text[-length:] == tag[:length]:
                return length
        return 0

    TAG = "<tool_call>"
    cases = [
        # (text, expected_len, description)
        ("hello world", 0, "no prefix"),
        ("hello <", 1, "just <"),
        ("hello <t", 2, "<t prefix"),
        ("hello <to", 3, "<to prefix"),
        ("hello <too", 4, "<too prefix"),
        ("hello <tool", 5, "<tool prefix"),
        ("hello <tool_", 6, "<tool_ prefix"),
        ("hello <tool_c", 7, "<tool_c prefix"),
        ("hello <tool_ca", 8, "<tool_ca prefix"),
        ("hello <tool_cal", 9, "<tool_cal prefix"),
        ("hello <tool_call", 10, "<tool_call prefix (missing >)"),
        ("hello <toast", 0, "<toast — not a prefix after <to"),
        ("hello <div>", 0, "HTML tag — not a prefix"),
        ("<", 1, "buffer is just <"),
        ("<tool_call>", 0, "full tag — should be caught by .find(), not prefix"),
        ("", 0, "empty string"),
        ("hello <ta", 0, "<ta — diverges at 3rd char"),
    ]

    all_pass = True
    for text, expected, desc in cases:
        result = _tag_prefix_len(text, TAG)
        status = "ok" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  [{status}] {desc}: _tag_prefix_len({text!r}) = {result} (expected {expected})")

    if all_pass:
        print(f"\n[PASS] All {len(cases)} cases passed")
    else:
        print(f"\n[FAIL] Some cases failed")
    return all_pass


if __name__ == '__main__':
    print(f"Testing against: {BASE_URL}")
    print(f"Model: {MODEL}")

    results = {}
    tests = [
        ("tag_prefix_detection", test_tag_prefix_detection),
        ("list_models", test_list_models),
        ("single_tool_call", test_single_tool_call),
        ("multiple_tools", test_multiple_tools),
        ("multi_turn", test_multi_turn),
        ("no_tool_needed", test_no_tool_needed),
        ("stream_tool_call", test_stream_tool_call),
        ("stream_no_tool_needed", test_stream_no_tool_needed),
        ("local_file_tool_call", test_local_file_tool_call),
        ("tool_choice_none", test_tool_choice_none),
        ("tool_choice_required", test_tool_choice_required),
        ("tool_choice_specific", test_tool_choice_specific),
        ("complex_parameter_types", test_complex_parameter_types),
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
