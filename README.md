# llm-speedtest

轻量、开源、零第三方依赖的**本地大模型推理速度测试工具**（Python 3.9+ 标准库实现）。

它直接对你本机已经运行起来的 OpenAI 兼容接口（llama.cpp、Strata、vLLM、SGLang、Ollama、LM Studio、MLX、Project Maya 等）执行标准化测速，并将**仅包含脱敏硬件规格与速度指标**的结果汇总到 [本地大模型配置选择器（itbanana.com/llm-hardware）](https://itbanana.com/llm-hardware)，帮助大家看到更真实的硬件实测速度。

---

## 快速开始

### 方式一：直接丢给你的 AI Agent 跑
把本仓库地址发给你的 AI（Claude Code / Codex / Cursor / OpenClaw 等），直接说：
> 帮我按这个仓库的 `SKILL.md` 测一下我本地正在跑的模型速度。

### 方式二：自己命令行运行
1. 先只测不传，预览完整输出：
```bash
python3 speedtest.py --dry-run
```
2. 确认没问题后上传（也可显式补充引擎、量化、内存类型等）：
```bash
python3 speedtest.py \
  --endpoint http://127.0.0.1:8080/v1 \
  --engine strata \
  --quant "GSQ-RCO IQ3_S" \
  --mtp true \
  --ram-type DDR4 \
  --ram-channels 2
```
默认会在终端打印完整 JSON 并询问确认；加 `--yes` 可跳过确认提示。

### 常用参数
| 参数 | 说明 |
|---|---|
| `--endpoint URL` | 本地 OpenAI 兼容地址（不填则自动探测 `8080`/`11434`/`1234`/`8000`/`30000`/`8081`） |
| `--ctx 1024,32768` | 要测试的输入上下文长度（逗号分隔，默认 `1024`） |
| `--gen-tokens 256` | 每轮生成 token 数（默认 `256`） |
| `--runs 3` | 每档上下文重复测速轮数，取中位数（默认 `3`） |
| `--engine` / `--quant` / `--mtp` | 推理引擎、量化版本、是否开启 MTP |
| `--ram-type` / `--ram-channels` | 内存代数（`DDR4`/`DDR5` 等）与通道数 |
| `--detect-only` | 仅打印自动识别的硬件规格并退出 |
| `--dry-run` | 仅本地测速并打印 JSON，不上传 |

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
