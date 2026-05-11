"""Quick streaming vision test."""
import argparse, base64, json, sys
from pathlib import Path
import requests

def test_streaming_vision(base_url, model, image_path):
    print(f"Testing STREAMING vision with {model}...")
    data_url = f"data:image/png;base64,{base64.b64encode(Path(image_path).read_bytes()).decode()}"
    
    payload = {
        "model": model, "stream": True, "max_tokens": 1000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "这张图片里有什么内容？用一句话回答。若无法获取图片信息，如实回答。"},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
        ]}]
    }
    
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=120)
    print(f"Status: {resp.status_code}")
    
    full = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]": break
            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content: full += content
            except: pass
    
    print(f"Streamed content ({len(full)} chars):\n{full[:500]}")
    
    vision_keywords = ["pyproject", "genai2openai", "flask", "toml"]
    found = [k for k in vision_keywords if k.lower() in full.lower()]
    no_vision = ["无法", "cannot", "can't see", "看不到"]
    not_found = [k for k in no_vision if k.lower() in full.lower()]
    
    if found: print(f"\n[PASS] Streaming vision works! Found: {found}")
    elif not_found: print(f"\n[FAIL] Streaming vision failed. Found: {not_found}")
    else: print(f"\n[UNCERTAIN] Cannot determine")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:5000")
    p.add_argument("--model", default="gpt-5.5")
    a = p.parse_args()
    test_streaming_vision(a.base_url, a.model, str(Path(__file__).resolve().parent / "test_vision.png"))
