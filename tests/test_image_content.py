"""
Multi-image & OpenAI content[] format 单元测试

用法: uv run tests/test_image_content.py

测试场景:
  1. _extract_text_from_content with plain string
  2. _extract_text_from_content with content[] array
  3. _extract_text_from_content with mixed text + images
  4. _extract_text_from_content with empty content
  5. _content_has_images with string content
  6. _content_has_images with image_url parts
  7. _content_has_images with text-only array
  8. _flatten_messages_text_only mixed messages
  9. convert_messages_to_genai_format
 10. convert_messages_to_genai_format (image-only)
 11. _content_has_images edge cases
 12. _content_has_images with input_image type
 13. _extract_text_from_content with input_text type
 14. _flatten_messages_text_only with input_image
 15. _coerce_content converts input_image -> image_url
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from provider.genai import (
    _extract_text_from_content,
    _content_has_images,
    _flatten_messages_text_only,
    convert_messages_to_genai_format,
)
from tools.responses.input import _coerce_content


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_extract_text_plain_string():
    print_separator("Test 1: extract text from plain string")
    result = _extract_text_from_content("Hello world")
    assert result == "Hello world", f"Expected 'Hello world', got {result!r}"
    print(f"  result: {result!r}")
    print("[PASS]")


def test_extract_text_content_array():
    print_separator("Test 2: extract text from content[] array")
    content = [
        {"type": "text", "text": "What's in this image?"},
    ]
    result = _extract_text_from_content(content)
    assert result == "What's in this image?", f"Expected text, got {result!r}"
    print(f"  result: {result!r}")
    print("[PASS]")


def test_extract_text_mixed():
    print_separator("Test 3: extract text from mixed text + image array")
    content = [
        {"type": "text", "text": "Compare image A"},
        {"type": "image_url", "image_url": {"url": "data:png;base64,abc", "detail": "auto"}},
        {"type": "text", "text": "and image B"},
        {"type": "image_url", "image_url": {"url": "data:png;base64,def", "detail": "auto"}},
    ]
    result = _extract_text_from_content(content)
    expected = "Compare image A and image B"
    assert result == expected, f"Expected {expected!r}, got {result!r}"
    print(f"  result: {result!r}")
    print("[PASS]")


def test_extract_text_empty():
    print_separator("Test 4: extract text from empty content")
    assert _extract_text_from_content("") == ""
    assert _extract_text_from_content([]) == ""
    assert _extract_text_from_content(None) == ""
    print("[PASS] all empty cases handled")


def test_has_images_string():
    print_separator("Test 5: _content_has_images with string")
    assert _content_has_images("hello") is False
    print("[PASS] string content has no images")


def test_has_images_true():
    print_separator("Test 6: _content_has_images with image_url parts")
    content = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:png;base64,abc"}},
    ]
    assert _content_has_images(content) is True
    print("[PASS] detected image_url in content array")


def test_has_images_text_only_array():
    print_separator("Test 7: _content_has_images with text-only array")
    content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    assert _content_has_images(content) is False
    print("[PASS] text-only array has no images")


def test_flatten_messages():
    print_separator("Test 8: _flatten_messages_text_only")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:png;base64,abc"}},
            {"type": "text", "text": "and that"},
        ]},
        {"role": "assistant", "content": "Sure!"},
        {"role": "user", "content": "plain text"},
    ]

    flattened = _flatten_messages_text_only(messages)
    assert len(flattened) == 4
    assert flattened[0]["content"] == "You are a helpful assistant."
    assert flattened[1]["content"] == "Describe this and that"
    assert flattened[1].get("role") == "user"
    assert flattened[2]["content"] == "Sure!"
    assert flattened[3]["content"] == "plain text"
    print(f"  flattened: {json.dumps(flattened, ensure_ascii=False)[:300]}")
    print("[PASS]")


def test_convert_messages():
    print_separator("Test 9: convert_messages_to_genai_format")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "text", "text": "What's in these images"},
            {"type": "image_url", "image_url": {"url": "data:png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:png;base64,def"}},
        ]},
    ]
    chat_info = convert_messages_to_genai_format(messages)
    expected = "What's in these images"
    assert chat_info == expected, f"Expected {expected!r}, got {chat_info!r}"
    print(f"  chat_info: {chat_info!r}")
    print("[PASS]")


def test_convert_messages_image_only():
    print_separator("Test 10: convert_messages_to_genai_format (image-only, no text)")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:png;base64,def"}},
        ]},
    ]
    chat_info = convert_messages_to_genai_format(messages)
    assert chat_info == "", f"Expected empty string for image-only, got {chat_info!r}"
    print(f"  chat_info: {chat_info!r} (empty - images only)")
    print("[PASS]")


def test_has_images_various_formats():
    print_separator("Test 11: _content_has_images edge cases")
    assert _content_has_images([]) is False
    assert _content_has_images([{"type": "image_url"}]) is True
    assert _content_has_images([{"type": "image_url", "image_url": {}}]) is True
    assert _content_has_images([{"type": "text", "text": ""}]) is False
    assert _content_has_images("") is False
    assert _content_has_images(None) is False
    print("[PASS] all edge cases handled")


def test_has_images_input_image():
    print_separator("Test 12: _content_has_images with input_image type")
    content = [
        {"type": "input_text", "text": "What's in this?"},
        {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
    ]
    assert _content_has_images(content) is True
    print("[PASS] detected input_image in content array")


def test_extract_text_input_text():
    print_separator("Test 13: _extract_text_from_content with input_text type")
    content = [
        {"type": "input_text", "text": "Hello"},
        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        {"type": "input_text", "text": "World"},
    ]
    result = _extract_text_from_content(content)
    assert result == "Hello World", f"Expected 'Hello World', got {result!r}"
    print(f"  result: {result!r}")
    print("[PASS]")


def test_flatten_messages_input_image():
    print_separator("Test 14: _flatten_messages_text_only with input_image")
    messages = [
        {"role": "user", "content": [
            {"type": "input_text", "text": "Describe"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ]},
    ]
    flattened = _flatten_messages_text_only(messages)
    assert flattened[0]["content"] == "Describe"
    print(f"  flattened content: {flattened[0]['content']!r}")
    print("[PASS]")


def test_coerce_content_input_image():
    print_separator("Test 15: _coerce_content converts input_image -> image_url")
    content = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
    ]
    result = _coerce_content(content)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

    text_part = result[0]
    assert text_part["type"] == "text", f"Expected text type, got {text_part['type']}"
    assert text_part["text"] == "What's in this image?"

    image_part = result[1]
    assert image_part["type"] == "image_url", f"Expected image_url type, got {image_part['type']}"
    assert image_part["image_url"]["url"] == "data:image/png;base64,abc"
    assert image_part["image_url"]["detail"] == "high"

    print(f"  converted: {json.dumps(result, ensure_ascii=False)[:300]}")
    print("[PASS] input_image converted to image_url format")


def test_coerce_content_text_only():
    print_separator("Test 16: _coerce_content text-only returns string")
    content = [
        {"type": "input_text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ]
    result = _coerce_content(content)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert result == "Hello\nWorld", f"Expected 'Hello\\nWorld', got {result!r}"
    print(f"  result: {result!r}")
    print("[PASS]")


if __name__ == "__main__":
    tests = [
        test_extract_text_plain_string,
        test_extract_text_content_array,
        test_extract_text_mixed,
        test_extract_text_empty,
        test_has_images_string,
        test_has_images_true,
        test_has_images_text_only_array,
        test_flatten_messages,
        test_convert_messages,
        test_convert_messages_image_only,
        test_has_images_various_formats,
        test_has_images_input_image,
        test_extract_text_input_text,
        test_flatten_messages_input_image,
        test_coerce_content_input_image,
        test_coerce_content_text_only,
    ]

    results = {}
    for fn in tests:
        try:
            fn()
            results[fn.__name__] = True
        except Exception as e:
            print(f"\n[FAIL] {fn.__name__}: {e}")
            results[fn.__name__] = False

    print_separator("Summary")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
