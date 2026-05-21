# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

OpenAI-compatible proxy in front of ShanghaiTech's GenAI platform. Translates OpenAI `/v1/chat/completions` and `/v1/responses` (plus `/v1/models`, `/health`) into upstream GenAI calls, then maps responses back. Tool calling, `tool_choice`, vision, and reasoning content are all implemented as a translation layer — the upstream does not natively speak OpenAI.

Python 3.11+, Flask, no DB. State for the Responses API is in-process only.

## Commands

```bash
uv sync                                    # install deps
uv run main.py --token "<sid>@<pw>"        # run (auto CAS login + refresh)
uv run main.py --token "eyJ..."            # run with raw JWT (no auto-refresh)
uv run main.py --token ... --debug         # verbose logging
uv run main.py --token ... --api-key K     # require client API key
uv run --env-file .env main.py --token ...  # load env vars from file

If `Config.model_mapping` is not set, no extra model aliases are added. Use `--config '{"mapping":{...}}'` to provide it.

uv run tests/run_offline.py                # all offline tests (no server needed)
uv run python tests/test_tool_calling.py   # single test file
bash tests/test_curl.sh [BASE_URL]         # smoke against a running server

docker compose -f docker-compose.local.yml up -d   # local build
TOKEN="..." docker compose up -d                   # prebuilt image
```

Online/integration tests (`test_responses.py`, `test_tool_calling.py`, `test_vision_*.py`, `test_reasoning_capture.py`, `test_errors.py`) need a live server; offline tests in `run_offline.py` mock the upstream.

## Architecture

Request flow: `main.py` → `app.create_app` → `api/*` route → adapter injects tool prompts into messages → `provider/genai.py` posts to upstream → response is parsed and re-shaped back into OpenAI format (streaming SSE or one-shot).

Key seams:

- **`auth/token_manager.py`** — accepts either a JWT or `student_id@password`. In credential mode it calls `auth/cas_login.py` (ShanghaiTech CAS) at startup and silently re-logs on expiry. `provider/genai.py` detects upstream "Token失效" markers and, when `fallback_renew` is on, asks the manager to renew and retries once. JWT mode has no renew path — expiry surfaces as 401.
- **`model_config/registry.py` + `spec.py`** — single source of truth mapping public model id → upstream `genai_id`, `root_ai_type` (`azure` / `xinference` / …), `tool_adapter` choice, and feature flags like `supports_reasoning`. `/v1/models` is also derived from here (intersected with what the upstream actually returns).
- **`tools/adapters/`** — `GenericAdapter` (JSON-style tool calls) and `GlmAdapter` (GLM-style) both subclass `ToolAdapter`. The adapter `inject()`s tool schemas + `tool_choice` rules into the system/user messages as prompts (the upstream models don't have native function calling), then `extract_tool_calls()` pulls them back out of the model's text output. Selection happens via `model_config` → `tools.adapters.get_adapter(name)`. Add a new model = add a registry entry; add a new tool-call wire format = new adapter class + register.
- **`api/chat.py`** — Chat Completions endpoint. Handles streaming vs non-streaming, reasoning_content splitting, vision image passthrough, and tool-call extraction.
- **`api/responses.py` + `tools/responses/state.py`** — Responses API. In-memory store keyed by `response_id` supports `previous_response_id` threading, `function_call_output` follow-ups, plus `retrieve` / `cancel`. State is **process-local**: a restart loses it, and multi-worker deployments will break linkage.
- **`errors.py`** — every error path goes through `openai_error()` / `make_error_chunk()` so clients always see OpenAI-shaped errors. New exceptions should be registered in `app.create_app` like `TokenExpiredError` / `LoginError`.

## Conventions worth knowing

- README, commit messages, and inline comments are predominantly Chinese — match the surrounding language when editing.
- Tool calling is **prompt-injected**, not native. When debugging a "model ignored my tool", look at the adapter's injected prompt first, then `extract_tool_calls()`.
- `provider/genai.py` is the only file that talks to the upstream; new endpoints should reuse its session, retry, and token-renew plumbing rather than calling `requests` directly.
- Streaming responses emit OpenAI-shaped SSE chunks; reasoning content (DeepSeek/o3) is surfaced as `reasoning_content` deltas separate from `content`.
