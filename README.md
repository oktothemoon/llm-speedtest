# llm-speedtest

轻量、开源、零第三方依赖的**本机大模型推理速度测试工具**（Python 3.9+ 标准库实现，支持 **Windows / macOS / Linux**）。

它直接对你**当前电脑**上已经运行起来的 OpenAI 兼容接口（llama.cpp、Strata、vLLM、SGLang、Ollama、LM Studio、MLX、Project Maya 等）执行标准化测速，自动识别并锁定本机硬件规格，并将**仅包含脱敏硬件规格与速度指标**的结果汇总到 [本地大模型速度天梯图（itbanana.com/llm-speed）](https://itbanana.com/llm-speed) 与 [本地大模型配置选择器（itbanana.com/llm-hardware）](https://itbanana.com/llm-hardware)，方便对比同款硬件的真实水平与调优上限。

> 为保证实测数据与硬件规格 100% 真实可信，本工具**仅支持在运行大模型的本机进行测速**（仅允许 `127.0.0.1` / `localhost`）。如果你的模型跑在另一台电脑或 Linux 服务器上，请直接到那台机器上（或通过 SSH 登录后）运行本脚本。

---

## 快速开始

### 方式一：自己直接运行（支持终端交互或打开本地 WebUI）

**macOS / Linux：**
```bash
git clone https://github.com/oktothemoon/llm-speedtest.git
cd llm-speedtest
python3 speedtest.py
```

**Windows（PowerShell / CMD）：**
```powershell
git clone https://github.com/oktothemoon/llm-speedtest.git
cd llm-speedtest
python speedtest.py
```

直接运行后会出现友好的中文引导菜单，你可以自由选择：
- **`[1]` 终端命令行交互测速（默认）**：适合本机或 SSH 远程终端直接操作，自动识别本机硬件、扫描本机端口与模型 → 测速 → 查看成绩单与明文 JSON → 确认上传后展示专属上榜战绩卡片与天梯直达链接。
- **`[2]` 打开本地 WebUI 可视化页面**：仅监听 `127.0.0.1`、自动在浏览器打开，点点鼠标即可完成测速与带礼花动效的上榜提交。

也可以通过参数直接进入对应模式：
- `python3 speedtest.py --cli`（或 Windows 下 `python speedtest.py --cli`）：直接进入终端交互模式
- `python3 speedtest.py --ui`：直接启动本地 WebUI 页面

### 方式二：直接丢给你的 AI Agent 跑（最省事）
直接把下面这段话复制发给你的 AI（Claude Code / Codex / Cursor / OpenClaw / Antigravity 等）：

```text
请先拉取或查看 https://github.com/oktothemoon/llm-speedtest 里的 SKILL.md，按里面的步骤帮我测一下我本机正在运行的本地大模型速度。先用 --non-interactive --dry-run 跑完，把成绩单和完整 JSON 发我确认后再决定是否上传。
```

---

## 进阶命令行参数（可选，主要供脚本与 AI 使用）
| 参数 | 说明 |
|---|---|
| `--cli` / `--ui` | 直接进入终端命令行模式 / 直接打开本地 WebUI 页面 |
| `--endpoint URL` | 本机 OpenAI 兼容地址或端口号（如 `8080` 或 `http://127.0.0.1:8080/v1`，不填则自动扫描） |
| `--ctx 1k,32k,64k,128k,256k` | 要测试的输入上下文档位，逗号分隔、支持 `k` 缩写（默认 `1k,32k,64k,128k,256k`，超限档位自动跳过） |
| `--gen-tokens 256` | 每轮生成 token 数（默认 `256`） |
| `--runs 3` | 短上下文每档测速轮数，取中位数（默认 `3`；≥64k 自动 1 轮） |
| `--engine` / `--quant` / `--kv-cache` / `--mtp` | 推理引擎、量化版本、KV Cache 精度（`f16`/`bf16`/`fp8`/`int8`/`q8_0`/`q4_0`）、是否开启 MTP |
| `--name` / `--note` | 选填：天梯图公开署名（≤20 字）与一句话调优备注（≤120 字），不填默认匿名 |
| `--ram-type` / `--ram-channels` | 内存代数（`DDR4`/`DDR5` 等）与通道数 |
| `--detect-only` | 仅打印本机自动识别的硬件规格并退出 |
| `--dry-run` / `--non-interactive` / `--yes` | 仅测不传 / 关闭交互向导 / 免确认直接上传 |

---

## 隐私说明（零个人信息，欢迎交给你的 AI 审计源码）

- **绝不采集**：用户名、主机名、本地文件路径、命令行参数、API Key、MAC 地址、GPU UUID、硬件序列号、环境变量。
- **上传前明文可见**：界面和终端都会完整展示即将发送的 JSON，默认需你手动确认才会上传。
- **服务端不保存原始 IP**：接收端仅使用加盐单向哈希做每小时限流与同机去重。
- **零第三方依赖**：全部代码只有 `speedtest.py` 和 `ui.py` 两个纯 Python 标准库文件，不放心可以直接丢给你的 AI 检查源码后再运行。

---

## 测试方法说明

- 先发 1 次短预热请求（不计入成绩），避免冷启动影响。
- 每轮 prompt 开头带有不同的随机轮次标识，防止引擎命中 KV 前缀缓存导致读题速度虚高。
- 优先读取引擎精确返回的 `timings`（`predicted_per_second` / `prompt_per_second`），同时独立记录客户端 SSE 流式测算的 `sse_tg_tps` 供服务端交叉校验。
- 自动识别常见魔改大显存卡（例如根据 `RTX 4090` + `48GB` 显存自动映射为 `rtx4090-48`）。

## 运行自测

```bash
python3 -m unittest test_speedtest.py
# Windows:
python -m unittest test_speedtest.py
```

## License

MIT
