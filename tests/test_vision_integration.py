"""
Vision image passthrough integration test for GenAI2OpenAI.

Tests image (test_vision.png - screenshot of pyproject.toml) is sent to gpt-5.5.
Prompt asks model to honestly report if it cannot get image info.
Usage: uv run tests/test_vision_integration.py [--base-url http://localhost:5000]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = path.read_bytes()
    mime = "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def test_vision(base_url: str, model: str, image_path: str):
    print(f"\n{'='*60}")
    print(f"  Vision Image Passthrough Test")
    print(f"  Model: {model}")
    print(f"  Image: {image_path}")
    print(f"{'='*60}\n")

    # Verify image exists
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[SKIP] Image not found: {image_path}")
        return

    data_url = encode_image(image_path)
    print(f"  Image encoded: {len(base64.b64encode(img_path.read_bytes()).decode('utf-8'))} base64 chars")
    print(f"  Image size: {img_path.stat().st_size} bytes\n")

    payload = {
        "model": model,
        "stream": False,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请仔细查看这张图片的内容，并用中文详细描述你看到的所有文字信息。"
                            "这是一张软件项目配置文件的截图。"
                            "如果你无法获取图片信息，请如实回答你无法看到图片内容。"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    }

    print("Sending request to:", f"{base_url}/v1/chat/completions")
    print("Request payload (text only):")
    text_content = payload["messages"][0]["content"][0]["text"]
    print(f"  {text_content[:200]}...\n")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120
        )
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to {base_url}. Is the server running?")
        sys.exit(1)

    print(f"HTTP Status: {resp.status_code}")
    print(f"Response headers: {dict(resp.headers)}\n")

    if resp.status_code != 200:
        print(f"[FAIL] Expected 200, got {resp.status_code}")
        print(f"Response body: {resp.text[:2000]}")
        return

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"[FAIL] Response is not valid JSON: {resp.text[:2000]}")
        return

    print("Parsed response:")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
    print()

    # Extract response content
    choices = data.get("choices", [])
    if not choices:
        print("[FAIL] No choices in response")
        return

    message = choices[0].get("message", {})
    content = message.get("content", "")
    finish_reason = choices[0].get("finish_reason", "N/A")

    print(f"Finish Reason: {finish_reason}")
    print(f"Response length: {len(content)} chars")
    print(f"\nModel response:\n{'-'*40}\n{content}\n{'-'*40}\n")

    # Check if model could see the image
    content_lower = content.lower()

    # Keywords that indicate model saw pyproject.toml content
    vision_indicators = [
        "pyproject", "toml", "[project]", "dependencies", "genai2openai",
        "flask", "python", "requires-python", ">=3.11", "requests",
        "pycryptodome", "openai", "shanghaitech",
    ]

    found_indicators = [kw for kw in vision_indicators if kw.lower() in content_lower]

    # Keywords that indicate model could NOT see image
    no_vision_indicators = [
        "无法", "cannot", "unable to", "don't have access", "sorry",
        "apologize", "unfortunately", "not able to see", "can't see",
        "没有收到", "没有获取", "没有看到", "看不到", "无法获取", "无法查看",
        "图片信息", "image", "没有提供",
    ]

    found_no_vision = [kw for kw in no_vision_indicators if kw.lower() in content_lower]

    print("Analysis:")
    print(f"  Vision-positive indicators found: {found_indicators}")
    print(f"  Vision-negative indicators found: {found_no_vision}")

    if found_indicators:
        print(f"\n[PASS] Model successfully read the image! Detected pyproject.toml content.")
    elif found_no_vision:
        print(f"\n[FAIL] Model could NOT read the image. Vision passthrough may be broken.")
        print("  Check if the upstream GenAI API supports image input for this model.")
    else:
        print(f"\n[UNCERTAIN] Could not determine if model saw the image.")
        print("  Response doesn't clearly indicate success or failure.")


def test_text_only_compare(base_url: str, model: str):
    """Bonus: send text-only request to compare behavior."""
    print(f"\n{'='*60}")
    print(f"  Text-Only Baseline Test (no image)")
    print(f"{'='*60}\n")

    payload = {
        "model": model,
        "stream": False,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": "请用中文回答：pyproject.toml 文件通常包含哪些内容？简要说明即可。"
            }
        ]
    }

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"Response:\n{content[:500]}\n")
        print("[PASS] Text-only request succeeded")
    else:
        print(f"[FAIL] Text-only request failed: {resp.status_code}")
        print(resp.text[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vision image passthrough")
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--model", default="gpt-5.5")
    args = parser.parse_args()

    image_path = str(Path(__file__).resolve().parent / "test_vision.png")

    test_text_only_compare(args.base_url, args.model)
    test_vision(args.base_url, args.model, image_path)
