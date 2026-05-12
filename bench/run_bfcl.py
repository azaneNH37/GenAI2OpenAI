"""轻量级 BFCL benchmark runner。

把 BFCL 数据集翻译成 OpenAI Chat Completions 请求，打到本地 GenAI2OpenAI 代理，
然后用 BFCL 自带的 ``possible_answer`` 做 AST-style 校验。专门用来对比每个 xinference
模型在**专属适配器**和 ``generic`` 适配器下的工具调用稳定性。

不依赖 BFCL 官方 handler 体系——后者要求实现一整套 BaseHandler 子类并且和
``bfcl_eval`` 包深度耦合，对我们这种"代理本身就在做格式转换"的场景没有意义。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
BFCL_DATA = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

# 数据集别名 -> (题目文件, 答案文件)
DATASETS: dict[str, tuple[str, str]] = {
    "simple_python": ("BFCL_v4_simple_python.json", "possible_answer/BFCL_v4_simple_python.json"),
    "live_simple": ("BFCL_v4_live_simple.json", "possible_answer/BFCL_v4_live_simple.json"),
    "java": ("BFCL_v4_simple_java.json", "possible_answer/BFCL_v4_simple_java.json"),
    "javascript": ("BFCL_v4_simple_javascript.json", "possible_answer/BFCL_v4_simple_javascript.json"),
}

DEFAULT_MODELS = "deepseek-v4-flash,deepseek-v4-pro,minimax-m1,glm-5.1,qwen-instruct"
DEFAULT_ADAPTERS = "auto,generic"


# ---------- BFCL → OpenAI 转换 ----------

def _bfcl_param_to_openai(schema: dict) -> dict:
    """BFCL 用 ``"type": "dict"`` 表示对象，用自定义类型如 ``tuple``/``any``，
    需要按 OpenAI / JSONSchema 标准翻译。"""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    t = out.get("type")
    type_map = {
        "dict": "object",
        "tuple": "array",
        "float": "number",
        "any": "string",
    }
    if t in type_map:
        out["type"] = type_map[t]
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _bfcl_param_to_openai(v) for k, v in out["properties"].items()}
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _bfcl_param_to_openai(out["items"])
    return out


def bfcl_to_openai_tools(functions: list[dict]) -> list[dict]:
    tools = []
    for fn in functions:
        params = _bfcl_param_to_openai(fn.get("parameters", {"type": "object", "properties": {}}))
        tools.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": params,
            },
        })
    return tools


def first_turn_messages(question: list[list[dict]]) -> list[dict]:
    # BFCL question 是 [[{role,content}, ...]]，simple 类只有一个 turn
    return list(question[0])


# ---------- 评分 ----------

def _values_for_param(allowed: list, value) -> bool:
    """BFCL 的 possible_answer 把每个参数的合法值列成 list。空字符串 ``""`` 表示
    "可缺省"。我们做宽松匹配：字符串大小写与空白不敏感。"""
    if not isinstance(allowed, list):
        allowed = [allowed]

    def norm(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    nv = norm(value)
    return any(norm(a) == nv for a in allowed)


def score_call(emitted: dict, ground_truth: list[dict]) -> str:
    """ground_truth 是 list of { fn_name: {arg: [allowed_values...]} }。
    匹配任一组合即 pass。"""
    if not emitted:
        return "no_tool_call"

    name = emitted["function"]["name"]
    try:
        args = json.loads(emitted["function"].get("arguments", "{}"))
    except (json.JSONDecodeError, ValueError):
        return "wrong_args"
    if not isinstance(args, dict):
        return "wrong_args"

    name_seen = False
    for option in ground_truth:
        if name not in option:
            continue
        name_seen = True
        spec = option[name]
        ok = True
        for arg_name, allowed in spec.items():
            # ""（空字符串）在 allowed 里 -> 该参数可省略
            optional = isinstance(allowed, list) and "" in allowed
            if arg_name not in args:
                if optional:
                    continue
                ok = False
                break
            if not _values_for_param(allowed, args[arg_name]):
                ok = False
                break
        # 多余参数：宽松，忽略
        if ok:
            return "pass"

    return "wrong_args" if name_seen else "wrong_name"


# ---------- HTTP ----------

@dataclass
class CaseResult:
    case_id: str
    model: str
    adapter: str
    status: str  # pass / wrong_args / wrong_name / no_tool_call / error
    elapsed: float
    detail: str = ""


def run_one(
    base_url: str,
    api_key: str | None,
    model: str,
    adapter: str,
    case: dict,
    answers: dict[str, list[dict]],
    timeout: float,
) -> CaseResult:
    case_id = case["id"]
    tools = bfcl_to_openai_tools(case["function"])
    messages = first_turn_messages(case["question"])
    target_model = model if adapter == "auto" else f"{model}@{adapter}"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": target_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "stream": False,
        "temperature": 0,
    }

    t0 = time.monotonic()
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        if r.status_code != 200:
            return CaseResult(case_id, model, adapter, "error", elapsed,
                              f"HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        choices = body.get("choices") or []
        if not choices:
            return CaseResult(case_id, model, adapter, "no_tool_call", elapsed, "no choices")
        msg = choices[0].get("message", {})
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return CaseResult(case_id, model, adapter, "no_tool_call", elapsed,
                              (msg.get("content") or "")[:200])
        gt = answers.get(case_id, [])
        if not gt:
            return CaseResult(case_id, model, adapter, "error", elapsed, "no ground truth")
        # 取第一个 tool_call 评分（simple 题目本身只期望一个调用）
        status = score_call(tcs[0], gt)
        return CaseResult(case_id, model, adapter, status, elapsed)
    except requests.RequestException as e:
        return CaseResult(case_id, model, adapter, "error", time.monotonic() - t0, str(e)[:200])


# ---------- runner ----------

def load_dataset(name: str, limit: int | None) -> tuple[list[dict], dict[str, list[dict]]]:
    data_file, ans_file = DATASETS[name]
    cases = [json.loads(line) for line in (BFCL_DATA / data_file).open()]
    answers = {}
    for line in (BFCL_DATA / ans_file).open():
        obj = json.loads(line)
        answers[obj["id"]] = obj.get("ground_truth", [])
    if limit:
        cases = cases[:limit]
    return cases, answers


@dataclass
class Tally:
    total: int = 0
    pass_: int = 0
    wrong_args: int = 0
    wrong_name: int = 0
    no_tool_call: int = 0
    error: int = 0
    elapsed_total: float = 0.0
    errors_sample: list[str] = field(default_factory=list)

    def add(self, r: CaseResult):
        self.total += 1
        self.elapsed_total += r.elapsed
        attr = "pass_" if r.status == "pass" else r.status
        setattr(self, attr, getattr(self, attr) + 1)
        if r.status in ("error", "no_tool_call", "wrong_args", "wrong_name") and len(self.errors_sample) < 5:
            self.errors_sample.append(f"[{r.status}] {r.case_id}: {r.detail[:120]}")

    @property
    def pass_rate(self) -> float:
        return self.pass_ / self.total if self.total else 0.0


def write_summary(out_dir: Path, dataset: str, tallies: dict[tuple[str, str], Tally]) -> Path:
    rows = []
    for (model, adapter), t in tallies.items():
        rows.append((model, adapter, t))
    rows.sort(key=lambda r: (r[0], r[1]))

    lines = [
        f"# BFCL benchmark — `{dataset}`",
        "",
        "| model | adapter | n | pass | wrong_args | wrong_name | no_tool_call | error | pass_rate | avg_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, adapter, t in rows:
        avg = (t.elapsed_total / t.total) if t.total else 0
        lines.append(
            f"| {model} | {adapter} | {t.total} | {t.pass_} | {t.wrong_args} | {t.wrong_name} | "
            f"{t.no_tool_call} | {t.error} | {t.pass_rate:.1%} | {avg:.2f} |"
        )

    # 每个模型对比 auto vs generic
    lines += ["", "## auto vs generic", ""]
    by_model: dict[str, dict[str, Tally]] = {}
    for (m, a), t in tallies.items():
        by_model.setdefault(m, {})[a] = t
    lines.append("| model | auto pass | generic pass | Δ |")
    lines.append("|---|---:|---:|---:|")
    for model, by_a in sorted(by_model.items()):
        auto = by_a.get("auto")
        gen = by_a.get("generic")
        if not auto or not gen:
            continue
        delta = auto.pass_rate - gen.pass_rate
        lines.append(f"| {model} | {auto.pass_rate:.1%} | {gen.pass_rate:.1%} | {delta:+.1%} |")

    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://localhost:5000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    p.add_argument("--models", default=DEFAULT_MODELS,
                   help="comma-separated GenAI public model ids")
    p.add_argument("--adapters", default=DEFAULT_ADAPTERS,
                   help="comma-separated adapter overrides; 'auto' = registry default")
    p.add_argument("--dataset", default="simple_python", choices=list(DATASETS))
    p.add_argument("--limit", type=int, default=50, help="0/negative = run full dataset")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default=None,
                   help="output JSON path; summary.md sits alongside")
    args = p.parse_args()

    if not BFCL_DATA.exists():
        sys.exit(f"BFCL data not found at {BFCL_DATA}. Run: git submodule update --init bench/gorilla")

    cases, answers = load_dataset(args.dataset, args.limit if args.limit > 0 else None)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"dataset={args.dataset} cases={len(cases)} models={models} adapters={adapters}")

    out_path = Path(args.out) if args.out else (
        ROOT / "results" / f"{args.dataset}-{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        (model, adapter, case)
        for model in models for adapter in adapters for case in cases
    ]
    print(f"total jobs: {len(jobs)}")

    results: list[CaseResult] = []
    tallies: dict[tuple[str, str], Tally] = {}

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_one, args.base_url, args.api_key, m, a, c, answers, args.timeout): (m, a, c["id"])
            for (m, a, c) in jobs
        }
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            tallies.setdefault((r.model, r.adapter), Tally()).add(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] last: {r.model}@{r.adapter} {r.case_id} -> {r.status} ({r.elapsed:.1f}s)")

    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "base_url": args.base_url,
        "models": models,
        "adapters": adapters,
        "n_cases": len(cases),
        "tallies": {f"{m}@{a}": asdict(t) for (m, a), t in tallies.items()},
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))

    summary = write_summary(out_path.parent, args.dataset, tallies)
    print(f"\nresults: {out_path}")
    print(f"summary: {summary}")
    print()
    print(summary.read_text())


if __name__ == "__main__":
    main()
