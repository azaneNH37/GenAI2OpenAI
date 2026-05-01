"""
Minimal reasoning_content capture test.
Usage:
  uv run tests/test_reasoning_capture.py --base-url http://localhost:5000 --model GPT-4.1
"""

import argparse
import json
import requests


def _print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_non_stream(base_url, model):
    _print_header("Non-stream reasoning_content")
    resp = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Solve 23*17. If supported, include reasoning in the response.",
            }
        ],
        "stream": False,
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content")
    if reasoning:
        print("\n[PASS] reasoning_content captured (non-stream)")
        print(f"  reasoning length: {len(reasoning)}")
        return True

    print("\n[WARN] reasoning_content not present in non-stream response")
    return False


def test_stream(base_url, model):
    _print_header("Stream reasoning_content")
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Solve 23*17. If supported, emit reasoning in deltas.",
                }
            ],
            "stream": True,
        },
        stream=True,
    )

    content_parts = []
    reasoning_parts = []

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)

    print(f"content length: {len(content)}")
    print(f"reasoning length: {len(reasoning)}")
    if reasoning:
        print("\n[PASS] reasoning_content captured (stream)")
        return True

    print("\n[WARN] reasoning_content not present in stream deltas")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--model", default="GPT-4.1")
    args = parser.parse_args()

    print(f"Testing against: {args.base_url}")
    print(f"Model: {args.model}")

    ok1 = test_non_stream(args.base_url, args.model)
    ok2 = test_stream(args.base_url, args.model)

    _print_header("Summary")
    print(f"  non-stream reasoning: {'PASS' if ok1 else 'WARN'}")
    print(f"  stream reasoning:     {'PASS' if ok2 else 'WARN'}")


if __name__ == "__main__":
    main()
