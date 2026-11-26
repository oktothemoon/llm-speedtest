# llm-speedtest — AI Agent 操作指南（SKILL）

如果你是用户的 AI Agent（如 Claude Code、Codex、Cursor、OpenClaw、Antigravity 等），用户把本仓库交给你跑本地大模型测速，请按以下 4 步执行：

## 第 1 步：探测运行中的推理服务与硬件配置
不要替用户下载模型，直接检查已经跑起来的 OpenAI 兼容服务：

- **情况 A：推理服务在本机（推荐）**
  1. 运行 `python3 speedtest.py --detect-only` 检查脚本自动识别的硬件规格（显卡型号、显存、内存）。
  2. 如用户没有告知端口，检查本机常见推理端口（`8080`、`11434`、`1234`、`8000`、`30000`、`8081`）或查看当前运行的推理进程名（如 `llama-server`、`strata`、`vllm`、`sglang`、`ollama`、`mlx`、`maya`）。
- **情况 B：推理服务在局域网另一台机器（如 `http://192.168.x.x:8080/v1`）**
  1. **优先尝试 SSH 到那台推理机上运行 `speedtest.py`**，这样硬件自动识别最准。
  2. 如果无法 SSH、只能在当前机器跨局域网测：`speedtest.py` 会**自动禁用本机硬件探测**（避免把笔记本配置错报成服务器配置）。你必须先向用户确认**推理机**的真实规格，并在命令里显式传入 `--hw-id`（或 `--hw-raw`）、`--gpu-count`、`--vram-gb`、`--ram-gb`、`--ram-type`、`--ram-channels`、`--os`。

从进程参数、启动命令或向用户确认中提取精确的元信息（**不要提取任何绝对路径、用户名或密钥**）：
- `--engine`：`strata` / `llamacpp` / `vllm` / `sglang` / `mlx` / `maya`
- `--quant`：如 `GSQ-RCO IQ3_S`、`Q4_K_M`、`UD-Q4_K_XL`、`FP8`、`NVFP4`
- `--mtp`：`true` 或 `false`（确认是否开启 MTP / 推测解码）
- `--ram-type` / `--ram-channels`：如 `DDR4`、`DDR5` 和通道数（若已知）
- `--fit`：`gpu`（全放显存）或 `hybrid`（显存 + 内存）

## 第 2 步：先跑一次 `--dry-run`（仅本地测速，不上传）
```bash
python3 speedtest.py \
  --endpoint http://127.0.0.1:8080/v1 \
  --engine strata \
  --quant "GSQ-RCO IQ3_S" \
  --mtp true \
  --ram-type DDR4 \
  --ram-channels 2 \
  --fit hybrid \
  --dry-run
```
如需加测长上下文，传 `--ctx 1024,32768`。

## 第 3 步：向用户展示明文 JSON 并征得确认
把终端输出的完整 JSON 展示给用户确认：
- 核对硬件型号（特别是魔改大显存卡，如 `rtx4090-48`、`rtx4080s-32`）、内存、量化与速度是否准确；
- 确认里面**不包含任何个人信息**（无用户名、主机名、文件路径、序列号、IP）。

## 第 4 步：用户同意后再上传
如果用户之前已明确授权自动上传，或在第 3 步确认同意，去掉 `--dry-run` 加上 `--yes` 执行上传（上传至 `https://itbanana.com/api/llm-speedtest/submit`）：
```bash
python3 speedtest.py ... --yes
```

## 隐私红线
- 绝对不要把本地路径、环境变量、主机名、用户名、API Key 放进任何参数中。
- 未经用户同意不要带 `--yes` 直接上传。
