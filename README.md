# llm-speedtest

轻量、开源、零第三方依赖的**本地大模型推理速度测试工具**（Python 3.9+ 标准库实现）。

它直接对你本机已经运行起来的 OpenAI 兼容接口（llama.cpp、Strata、vLLM、SGLang、Ollama、LM Studio、MLX、Project Maya 等）执行标准化测速，并将**仅包含脱敏硬件规格与速度指标**的结果汇总到 [本地大模型配置选择器（itbanana.com/llm-hardware）](https://itbanana.com/llm-hardware)，帮助大家看到更真实的硬件实测速度。

---

## 快速开始

### 方式一：直接丢给你的 AI Agent 跑（最省事）
直接把下面这段话复制发给你的 AI（Claude Code / Codex / Cursor / OpenClaw 等）：

```text
请先拉取或查看 https://github.com/oktothemoon/llm-speedtest 里的 SKILL.md，按里面的步骤帮我测一下我本地（或局域网）正在运行的本地大模型速度。先用 --non-interactive --dry-run 跑完，把成绩单和完整 JSON 发我确认后再决定是否上传。
```

### 方式二：自己直接运行（极简交互，基本只需回车）
```bash
git clone https://github.com/oktothemoon/llm-speedtest.git
cd llm-speedtest
python3 speedtest.py
```
- **在本机跑**：自动识别显卡、内存、端口、模型与量化 → 按一下回车直接开始测速 → 测完看成绩单和明文 JSON，按 `y` 上传。
- **测局域网另一台电脑**：直接跑 `python3 speedtest.py --endpoint <局域网IP或地址>`（只输纯 IP 也行，脚本会自动试常见端口和路径），然后**用一句话说清推理机配置**即可，例如：
  - `4090 48G + 64G DDR4 Strata`
  - `双 7900 XTX 128G DDR5`
  - `M5 Ultra 256G`
  - 或者填 `user@局域网IP`（免密 SSH 自动识别远端硬件，不会上传 SSH 地址）

只想本地测完看看不上传，加 `--dry-run`：
```bash
python3 speedtest.py --dry-run
```

> 脚本检测到非本机地址时会自动关闭本机硬件探测，绝不会把你手里笔记本的配置错记成推理机配置。

### 进阶命令行参数（可选，主要供脚本与 AI 使用）
| 参数 | 说明 |
|---|---|
| `--endpoint URL` | OpenAI 兼容地址（不填则自动探测本机常见端口） |
| `--ssh user@host` | 局域网测速时通过只读 SSH 自动探测远端硬件（不会上传） |
| `--ctx 1024,32768` | 要测试的输入上下文长度（逗号分隔，默认 `1024`） |
| `--gen-tokens 256` | 每轮生成 token 数（默认 `256`） |
| `--runs 3` | 每档上下文重复测速轮数，取中位数（默认 `3`） |
| `--engine` / `--quant` / `--mtp` | 推理引擎、量化版本、是否开启 MTP |
| `--hw-id` / `--hw-raw` / `--vram-gb` / `--os` | 手动指定目标机硬件与系统 |
| `--ram-gb` / `--ram-type` / `--ram-channels` | 内存容量、代数（`DDR4`/`DDR5` 等）与通道数 |
| `--detect-only` | 仅打印本机自动识别的硬件规格并退出 |
| `--dry-run` / `--non-interactive` / `--yes` | 仅测不传 / 关闭交互向导 / 免确认直接上传 |

---

## 隐私说明（零个人信息）

- **绝不采集**：用户名、主机名、本地文件路径、命令行参数、MAC 地址、GPU UUID、硬件序列号、环境变量。
- **上传前明文可见**：脚本会在终端完整打印将要发送的 JSON，默认需你手动输入 `y` 才发送。
- **服务端不保存原始 IP**：接收端仅使用加盐单向哈希做每小时限流与同机去重。
- **零第三方依赖**：全部代码都在单文件 `speedtest.py` 中，几分钟就能读完审查。

---

## 测试方法说明

- 先发 1 次短预热请求（不计入成绩），避免冷启动影响。
- 每轮 prompt 开头带有不同的随机轮次标识，防止引擎命中 KV 前缀缓存导致读题速度虚高。
- 优先读取引擎精确返回的 `timings`（`predicted_per_second` / `prompt_per_second`），同时独立记录客户端 SSE 流式测算的 `sse_tg_tps` 供服务端交叉校验。
- 自动识别常见魔改大显存卡（例如根据 `RTX 4090` + `48GB` 显存自动映射为 `rtx4090-48`）。

## 运行自测

```bash
python3 -m unittest test_speedtest.py
```

## License

MIT
