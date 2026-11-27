# llm-speedtest — AI Agent 操作指南（SKILL）

如果你是用户的 AI Agent（如 Claude Code、Codex、Cursor、OpenClaw、Antigravity 等），用户把本仓库交给你跑本地大模型测速，请按以下 4 步执行（支持 Windows / macOS / Linux；Windows 上请将 `python3` 替换为 `python`）：

> **本机测速原则**：本工具仅支持在运行推理服务的**本机**测速（仅允许 `127.0.0.1` / `localhost`），不支持跨局域网远端测速。如果用户的推理服务运行在另一台服务器上，请先通过 SSH 登录到该服务器后再运行 `speedtest.py`。

## 第 1 步：探测本机运行中的推理服务与硬件配置
不要替用户下载模型，直接检查本机已经跑起来的 OpenAI 兼容服务：

1. 运行 `python3 speedtest.py --detect-only` 检查脚本自动识别的本机硬件规格（显卡型号、显存、内存）。
2. 如用户没有告知端口，检查本机常见推理端口（`8080`、`11434`、`1234`、`8000`、`30000`、`8081`）或查看当前运行的推理进程名（如 `llama-server`、`strata`、`vllm`、`sglang`、`ollama`、`mlx`、`maya`）。

从进程参数、启动命令或向用户确认中提取精确的元信息（**不要提取任何绝对路径、用户名或密钥**）：
- `--engine`：`strata` / `llamacpp` / `vllm` / `sglang` / `mlx` / `maya`
- `--quant`：如 `GSQ-RCO IQ3_S`、`Q4_K_M`、`UD-Q4_K_XL`、`FP8`、`NVFP4`
- `--kv-cache`：KV Cache 精度，如 `f16`、`bf16`、`fp8`、`int8`、`q8_0`、`q4_0`（从启动参数如 `-ctk`/`--cache-type-k`/`--kv-cache-dtype` 或向用户确认）
- `--mtp`：`true` 或 `false`（确认是否开启 MTP / 推测解码）
- `--ram-type` / `--ram-channels`：如 `DDR4`、`DDR5` 和通道数（若已知）
- `--fit`：`gpu`（全放显存）或 `hybrid`（显存 + 内存）
- `--name` / `--note`（完全可选）：若用户想在 [速度天梯图（itbanana.com/llm-speed）](https://itbanana.com/llm-speed) 上署名或留下调优参数备注，可传入自定义昵称（≤20 字）与调优说明（≤120 字，勿包含网址或个人信息）；用户未提供则留空保持匿名。
- `--anonymous` / `--client-id`（可选）：默认会自动附带本地随机匿名 `client_id`（保存在 `~/.config/llm-speedtest/client_id`，不含任何机器或个人隐私），便于日后升级硬件关联历史成绩；若用户希望完全不附带，可加 `--anonymous`。

## 第 2 步：先跑一次 `--non-interactive --dry-run`（仅本地测速，不上传）
默认会依次测试 `1k / 32k / 64k / 128k / 256k` 全套上下文档位（自动探测服务端上限，超出上限的档位自动跳过）。AI Agent 执行时请务必带上 `--non-interactive`（避免触发终端交互问答卡住）：
```bash
python3 speedtest.py \
  --endpoint http://127.0.0.1:8080/v1 \
  --engine strata \
  --quant "GSQ-RCO IQ3_S" \
  --kv-cache int8 \
  --mtp true \
  --ram-type DDR4 \
  --ram-channels 2 \
  --fit hybrid \
  --non-interactive \
  --dry-run
```
如需指定部分档位，传 `--ctx 1k,32k,64k,128k,256k`（或仅快速验证 `--ctx 1k`）。

## 第 3 步：向用户展示明文 JSON 并征得确认
把终端输出的完整 JSON 展示给用户确认：
- 核对硬件型号（特别是魔改大显存卡，如 `rtx4090-48`、`rtx4080s-32`）、内存、量化与速度是否准确；
- 确认里面**不包含任何个人信息**（无用户名、主机名、文件路径、硬件序列号、IP；`client_id` 仅为本地纯随机 UUID）。

## 第 4 步：用户同意后再上传
如果用户之前已明确授权自动上传，或在第 3 步确认同意，去掉 `--dry-run` 加上 `--yes` 执行上传（上传至 `https://itbanana.com/api/llm-speedtest/submit`）：
```bash
python3 speedtest.py ... --yes
```

## 隐私红线
- 绝对不要把本地路径、环境变量、主机名、用户名、API Key 放进任何参数中。
- 未经用户同意不要带 `--yes` 直接上传。
