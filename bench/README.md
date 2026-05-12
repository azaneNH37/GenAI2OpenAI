# bench/

工具调用稳定性 benchmark：用 BFCL（Berkeley Function Calling Leaderboard）的测试集
对比 GenAI 平台上各 xinference 模型，在**专属适配器** vs **generic 适配器**下的稳定性。

## 目录

- `gorilla/` — submodule，[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)，
  数据来源是 `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`
  及对应的 `possible_answer/`。
- `run_bfcl.py` — 轻量 runner，**不依赖 BFCL 的官方 handler 体系**，直接把 BFCL 题目
  翻译成 OpenAI Chat Completions 请求打到本地代理，再用 BFCL 的 `possible_answer`
  做 AST-style 校验。
- `results/` — 运行结果 (`*.json` + `summary.md`)。

## 用法

启动代理（确保监听本地端口）：

```bash
uv run main.py --token "<sid>@<pw>" --port 5000
```

跑 benchmark：

```bash
# 默认：simple_python 全集 × 几个 xinference 模型 × {专属适配器, generic}
uv run python bench/run_bfcl.py

# 自定义
uv run python bench/run_bfcl.py \
    --base-url http://localhost:5000 \
    --models deepseek-v4-flash,minimax-m1,glm-5.1,qwen-instruct \
    --adapters auto,generic \
    --dataset simple_python \
    --limit 50 \
    --concurrency 4 \
    --out bench/results/run-$(date +%Y%m%d-%H%M).json
```

`--adapters` 中的 `auto` 表示走模型在 `model_config/registry.py` 中声明的默认适配器；
其它值（`generic` / `glm` / `minimax` / `deepseek_v4` / `deepseek_legacy`）会通过
`model@adapter` 后缀强制覆盖（chat.py 里实现）。

## 评分

对每条样本：

| 状态 | 含义 |
|---|---|
| `pass` | 模型发出至少一个 tool_call，函数名正确，所有 required 参数齐全且值在 `possible_answer` 允许集合内 |
| `wrong_args` | 函数名对，但某个参数缺失或值不在允许集合内 |
| `wrong_name` | 调出了 tool_call，但函数名不对 |
| `no_tool_call` | 模型没有发出 tool_call（生成了纯文本） |
| `error` | HTTP / 解析异常 |

## 子模块管理

```bash
git submodule update --init --recursive bench/gorilla
```
