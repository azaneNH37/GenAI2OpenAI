"""
20 轮工具调用压力测试 — 验证 generic / glm 两套解析器在长对话中的稳定性

针对两种 tool_adapter:
  - generic adapter: 使用 deepseek-v4-flash (deepseek-chat)
  - glm adapter:     使用 glm-5.1 (chatglm)

每轮一次 user 提问 -> 期望模型产出 tool_call -> 回灌 tool_result -> 继续下一轮，
对话历史持续累加，模拟真实 agent 场景。

统计指标:
  - tool_call 解析成功率 (parse rate)
  - 工具名匹配率 (是否选到了期望的 tool)
  - arguments JSON 是否合法、是否包含必填字段
  - 不应出现的 raw <tool_call> 文本残留
  - 平均轮次延迟

用法:
  # 默认两个模型各跑 20 轮
  uv run tests/test_tool_calling_20rounds.py

  # 自定义
  uv run tests/test_tool_calling_20rounds.py --base-url http://localhost:5000 --rounds 20 \\
      --models deepseek-v4-flash glm-5.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))


CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_city",
        "description": "Return basic info (population, country) for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "include_population": {
                    "type": "boolean",
                    "description": "Whether to include population",
                },
            },
            "required": ["city"],
        },
    },
}

CONVERT_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_units",
        "description": "Convert a value from one unit to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric value"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
}

TOOLS = [CALCULATE_TOOL, LOOKUP_TOOL, CONVERT_TOOL]


# 每轮 (user prompt, expected_tool_name, fake_tool_result)
SCENARIOS = [
    ("What is 17 * 23?",                      "calculate",     {"result": 391}),
    ("Population of Tokyo?",                  "lookup_city",   {"city": "Tokyo", "population": 13960000, "country": "Japan"}),
    ("Convert 100 km to miles.",              "convert_units", {"result": 62.137}),
    ("Compute 2^10.",                         "calculate",     {"result": 1024}),
    ("Tell me about Paris.",                  "lookup_city",   {"city": "Paris", "population": 2161000, "country": "France"}),
    ("How much is 5 kg in pounds?",           "convert_units", {"result": 11.023}),
    ("What is 1234 + 5678?",                  "calculate",     {"result": 6912}),
    ("Lookup info on Cairo.",                 "lookup_city",   {"city": "Cairo", "population": 9540000, "country": "Egypt"}),
    ("Convert 32 fahrenheit to celsius.",     "convert_units", {"result": 0}),
    ("Calculate (45 + 55) / 2.",              "calculate",     {"result": 50}),
    ("Get info about Sydney.",                "lookup_city",   {"city": "Sydney", "population": 5312000, "country": "Australia"}),
    ("Convert 1 mile to meters.",             "convert_units", {"result": 1609.34}),
    ("What's 99 * 99?",                       "calculate",     {"result": 9801}),
    ("Population of Mumbai?",                 "lookup_city",   {"city": "Mumbai", "population": 20410000, "country": "India"}),
    ("Convert 50 mph to kph.",                "convert_units", {"result": 80.467}),
    ("Compute 144 / 12.",                     "calculate",     {"result": 12}),
    ("Tell me about Berlin.",                 "lookup_city",   {"city": "Berlin", "population": 3669000, "country": "Germany"}),
    ("Convert 10 liters to gallons.",         "convert_units", {"result": 2.642}),
    ("What is 7 * 8 + 3?",                    "calculate",     {"result": 59}),
    ("Lookup info on Toronto.",               "lookup_city",   {"city": "Toronto", "population": 2930000, "country": "Canada"}),
]


def _required_fields(tool_name):
    for tool in TOOLS:
        f = tool["function"]
        if f["name"] == tool_name:
            return f.get("parameters", {}).get("required", [])
    return []


def run_model(base_url, model, rounds):
    print(f"\n{'='*70}\n  Model: {model}  ({rounds} rounds)\n{'='*70}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. ALWAYS use the provided tools to "
                "answer factual / numeric questions instead of guessing. Emit "
                "exactly one tool_call per user question."
            ),
        }
    ]

    stats = {
        "rounds": 0,
        "tool_call_emitted": 0,
        "name_matched": 0,
        "args_json_valid": 0,
        "args_required_present": 0,
        "raw_tag_leak": 0,   # <tool_call> 文本未被解析
        "errors": 0,
        "latencies": [],
    }

    for i in range(rounds):
        prompt, expected_tool, fake_result = SCENARIOS[i % len(SCENARIOS)]
        stats["rounds"] += 1
        messages.append({"role": "user", "content": prompt})

        print(f"\n--- Round {i+1}/{rounds}: {prompt!r} (expect {expected_tool})")

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages, "tools": TOOLS, "stream": False},
                timeout=120,
            )
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            print(f"  [ERROR] HTTP/JSON: {e}")
            # 撤回这条 user message，避免脏对话
            messages.pop()
            continue
        finally:
            stats["latencies"].append(time.monotonic() - t0)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        content_text = msg.get("content") or ""

        if "<tool_call>" in content_text:
            stats["raw_tag_leak"] += 1
            print(f"  [WARN] raw <tool_call> tag leaked into content")

        if not tool_calls:
            print(f"  [FAIL] no tool_calls. content[:120]={content_text[:120]!r}")
            # 让对话继续：append 模型的回答，进入下一轮
            messages.append({"role": "assistant", "content": content_text or ""})
            continue

        stats["tool_call_emitted"] += 1
        tc = tool_calls[0]
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")

        if name == expected_tool:
            stats["name_matched"] += 1

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            stats["args_json_valid"] += 1
        except json.JSONDecodeError as e:
            args = {}
            print(f"  [FAIL] arguments not JSON: {e}; raw={args_raw[:120]!r}")

        required = _required_fields(name)
        missing = [k for k in required if k not in args]
        if not missing:
            stats["args_required_present"] += 1
        else:
            print(f"  [WARN] missing required {missing}; got keys={list(args.keys())}")

        print(f"  [OK] {name}({args_raw[:120]})")

        # 把 assistant tool_call + tool result 加入历史，进入下一轮
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id", "call_0"),
                "content": json.dumps(fake_result, ensure_ascii=False),
            }
        )

    return stats


def print_summary(model, stats):
    n = max(stats["rounds"], 1)
    avg_lat = sum(stats["latencies"]) / max(len(stats["latencies"]), 1)
    print(f"\n--- Summary [{model}] ---")
    print(f"  rounds:                 {stats['rounds']}")
    print(f"  tool_call emitted:      {stats['tool_call_emitted']:>3}/{n}  ({stats['tool_call_emitted']/n:.0%})")
    print(f"  name matched expected:  {stats['name_matched']:>3}/{n}  ({stats['name_matched']/n:.0%})")
    print(f"  arguments JSON valid:   {stats['args_json_valid']:>3}/{n}  ({stats['args_json_valid']/n:.0%})")
    print(f"  required fields ok:     {stats['args_required_present']:>3}/{n}  ({stats['args_required_present']/n:.0%})")
    print(f"  raw <tool_call> leak:   {stats['raw_tag_leak']}")
    print(f"  http/json errors:       {stats['errors']}")
    print(f"  avg latency:            {avg_lat:.2f}s")


def passed(stats, parse_threshold=0.9):
    """解析层判定: 模型只要 emit 了 tool_call, 解析必须 100% 成功;
    name match 用作弱信号, 不计 fail。"""
    emitted = stats["tool_call_emitted"]
    if emitted == 0:
        return False
    parse_ok = (
        stats["args_json_valid"] == emitted
        and stats["raw_tag_leak"] == 0
    )
    emit_rate = emitted / max(stats["rounds"], 1)
    return parse_ok and emit_rate >= parse_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:5000")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument(
        "--models",
        nargs="+",
        default=["deepseek-v4-flash", "glm-5.1"],
        help="Model ids to test (one per adapter)",
    )
    args = p.parse_args()

    print(f"Base URL : {args.base_url}")
    print(f"Models   : {args.models}")
    print(f"Rounds   : {args.rounds}")

    all_results = {}
    for model in args.models:
        stats = run_model(args.base_url, model, args.rounds)
        all_results[model] = stats
        print_summary(model, stats)

    print(f"\n{'='*70}\n  FINAL\n{'='*70}")
    overall_ok = True
    for model, stats in all_results.items():
        ok = passed(stats)
        overall_ok = overall_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {model}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
