"""
统一运行所有离线测试（不需要服务在跑）。
用法: uv run tests/run_offline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

OFFLINE_TESTS = [
    "test_genai_retry.py",
    "test_genai_timeout.py",
    "test_image_content.py",
    "test_token_fallback_renew.py",
    "test_usage_accounting.py",
]


def main():
    failures = []
    for name in OFFLINE_TESTS:
        path = ROOT / name
        print(f"\n{'=' * 60}\n  Running {name}\n{'=' * 60}")
        result = subprocess.run([PY, str(path)], cwd=ROOT.parent)
        if result.returncode != 0:
            failures.append(name)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  FAIL: {len(failures)}/{len(OFFLINE_TESTS)} test files failed:")
        for name in failures:
            print(f"    - {name}")
        sys.exit(1)
    print(f"  OK: all {len(OFFLINE_TESTS)} offline test files passed")


if __name__ == "__main__":
    main()
