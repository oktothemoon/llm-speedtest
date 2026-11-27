#!/usr/bin/env python3
"""
llm-speedtest: Standardized, privacy-first speed test for local LLMs.
Zero external dependencies (Python 3.9+ stdlib only).

Measures prefill (pp) and token generation (tg) speed against an already-running
local OpenAI-compatible inference server (llama.cpp, Strata, vLLM, SGLang,
Ollama, LM Studio, MLX, Project Maya, etc.), prints the full desensitized JSON
payload, and optionally uploads it to https://itbanana.com/api/llm-speedtest/submit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

CLIENT_VERSION = "0.3.23"
USER_AGENT = f"llm-speedtest/{CLIENT_VERSION} (+https://github.com/oktothemoon/llm-speedtest)"
SCHEMA_VERSION = 1
SHOW_LAUNCH_MENU = False  # 「终端 / WebUI」启动选单；暂时关闭
DEFAULT_SUBMIT_URL = "https://itbanana.com/api/llm-speedtest/submit"
DEFAULT_STATS_URL = "https://itbanana.com/api/llm-speedtest/stats"

DEFAULT_CTX_TIERS = [1024, 32768, 65536, 131072, 262144]
DEFAULT_CTX_STR = "1k,32k,64k,128k,256k"
CLIENT_ID_RE = re.compile(r"^[0-9a-fA-F-]{16,64}$")
# 与 itbanana.com 后端 LADDER_MIN_GEN_TOKENS 保持一致：生成 token 数低于此值的成绩不计入天梯
LADDER_MIN_GEN_TOKENS = 64
# 记录不接受 ignore_eos 参数的本机服务（首次 400/422 后自动去掉该参数重试，后续请求不再携带）
_NO_IGNORE_EOS_ENDPOINTS: set = set()
# endpoint -> Ollama 原生 API 根地址（None 表示不是 Ollama 或原生接口不可用）
_OLLAMA_NATIVE_ROOTS: Dict[str, Optional[str]] = {}


def clean_client_id(val: Optional[str]) -> Optional[str]:
    """Validate an anonymous random client_id string (hex/UUID-like, no personal info)."""
    if not val:
        return None
    s = str(val).strip().lower()
    return s if CLIENT_ID_RE.fullmatch(s) else None


def get_client_id_path() -> str:
    """Return local file path for persisting the random anonymous client_id."""
    override = os.environ.get("LLM_SPEEDTEST_CLIENT_ID_FILE")
    if override:
        return os.path.expanduser(override)
    if platform.system().lower().startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "llm-speedtest", "client_id")


def get_or_create_client_id() -> str:
    """Load or create a purely random local anonymous UUIDv4 for hardware-upgrade continuity."""
    path = get_client_id_path()
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = clean_client_id(f.read())
            if existing:
                return existing
    except OSError:
        pass

    new_id = str(uuid.uuid4())
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_id + "\n")
        if os.name == "posix":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    except OSError:
        pass
    return new_id


def parse_ctx_list(ctx_str: Optional[str]) -> List[int]:
    """Parse comma-separated context sizes supporting '1k', '32k', '64k', '128k', '256k', or raw ints."""
    raw = (ctx_str or "").strip()
    if not raw:
        return list(DEFAULT_CTX_TIERS)
    out: List[int] = []
    for part in re.split(r"[,，\s]+", raw):
        s = part.strip().lower()
        if not s:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)k", s)
        if m:
            val = max(64, int(round(float(m.group(1)) * 1024)))
        elif s.isdigit():
            val = max(64, int(s))
        else:
            continue
        if val not in out:
            out.append(val)
    return out or list(DEFAULT_CTX_TIERS)


def format_ctx_label(tokens: int) -> str:
    for tier in DEFAULT_CTX_TIERS:
        if abs(tokens - tier) / float(tier) <= 0.04:
            return f"{tier // 1024}k"
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    if tokens >= 1000:
        k = tokens / 1024.0
        if abs(k - round(k)) < 0.08 and round(k) >= 1:
            return f"{int(round(k))}k"
    return str(tokens)


def effective_runs_for_ctx(ctx_prompt: int, base_runs: int) -> int:
    """Keep short contexts multi-run for stability while avoiding 15-minute waits on 64k–256k."""
    base = max(1, int(base_runs or 1))
    if ctx_prompt >= 65536:
        return 1
    if ctx_prompt >= 16384:
        return min(base, 2)
    return base


def timeout_for_ctx(ctx_prompt: int) -> float:
    if ctx_prompt >= 131072:
        return 900.0
    if ctx_prompt >= 65536:
        return 600.0
    if ctx_prompt >= 16384:
        return 420.0
    return 300.0


def normalize_kv_cache(val: Optional[str]) -> Optional[str]:
    """Normalize KV cache quantization label (e.g. f16, bf16, fp8, int8, q8_0, q4_0, int4)."""
    if not val:
        return None
    s = re.sub(r"\s+", "", str(val).strip().lower())
    if not s:
        return None
    aliases = {
        "fp16": "f16",
        "float16": "f16",
        "f16": "f16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp8": "fp8",
        "f8": "fp8",
        "fp8_e4m3": "fp8",
        "fp8_e5m2": "fp8",
        "int8": "int8",
        "q8": "q8_0",
        "q8_0": "q8_0",
        "q80": "q8_0",
        "q4": "q4_0",
        "q4_0": "q4_0",
        "q40": "q4_0",
        "q4_1": "q4_1",
        "int4": "int4",
        "q5_0": "q5_0",
        "q5_1": "q5_1",
        "f32": "f32",
    }
    if s in aliases:
        return aliases[s]
    # Support asymmetric K/V notation like q8_0/q4_0
    if "/" in s:
        parts = [aliases.get(p, p) for p in s.split("/", 1)]
        return "/".join(parts)[:24]
    return s[:24]

_ANSI_ENABLED: Optional[bool] = None


def configure_console() -> None:
    """Ensure UTF-8 output, readline line-editing on POSIX, and ANSI color support."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if os.name != "nt":
        try:
            import readline  # noqa: F401 - enables arrow/delete keys in input() on Linux/macOS
        except Exception:
            pass


def _supports_color() -> bool:
    global _ANSI_ENABLED
    if _ANSI_ENABLED is not None:
        return _ANSI_ENABLED
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        _ANSI_ENABLED = False
        return False
    if not sys.stderr.isatty():
        _ANSI_ENABLED = False
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                    _ANSI_ENABLED = True
                    return True
        except Exception:
            pass
        _ANSI_ENABLED = bool(os.environ.get("WT_SESSION") or os.environ.get("ANSICON"))
        return _ANSI_ENABLED
    _ANSI_ENABLED = True
    return True


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z~]")


def strip_ansi(text: str) -> str:
    """Remove terminal color codes (for log lines shown in the WebUI)."""
    return _ANSI_RE.sub("", str(text))


def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _terminal_columns(default: int = 120) -> int:
    """Width of the terminal the card is printed to (stderr first, since stdout may be redirected)."""
    for stream in (sys.stderr, sys.stdout):
        try:
            return max(40, os.get_terminal_size(stream.fileno()).columns)
        except (OSError, ValueError, AttributeError):
            continue
    try:
        return max(40, int(os.environ.get("COLUMNS", default)))
    except ValueError:
        return default


_BREAK_AFTER = set(" ·，、|；：）)")


def _wrap_display(line: str, width: int, indent: str = "     ") -> List[str]:
    """Wrap a line (may contain ANSI colors, CJK) so each piece fits in `width` display columns.

    Prefers breaking after a space / "·" / "，" in the second half of the line; continuation
    lines are prefixed with `indent`.
    """
    if width <= 10 or _display_width(line) <= width:
        return [line]
    tokens = re.findall(r"\x1b\[[0-9;?]*[A-Za-z~]|.", line, flags=re.S)
    out: List[str] = []
    cur: List[str] = []
    cur_w = 0
    last_break = -1  # index in cur after which we may break
    prefix = ""
    prefix_w = 0
    for tok in tokens:
        if tok.startswith("\x1b"):
            cur.append(tok)
            continue
        w = 0 if unicodedata.combining(tok) else (2 if unicodedata.east_asian_width(tok) in ("W", "F") else 1)
        if cur_w + w > width - prefix_w and cur:
            if last_break >= 0 and _display_width("".join(cur[: last_break + 1])) >= (width - prefix_w) // 2:
                head, rest = cur[: last_break + 1], cur[last_break + 1 :]
            else:
                head, rest = cur, []
            out.append(prefix + "".join(head).rstrip())
            prefix, prefix_w = indent, _display_width(indent)
            # 续行开头不留空格
            while rest and rest[0] == " ":
                rest.pop(0)
            cur, cur_w, last_break = rest, _display_width("".join(rest)), -1
            for i, t in enumerate(cur):
                if t in _BREAK_AFTER:
                    last_break = i
        cur.append(tok)
        cur_w += w
        if tok in _BREAK_AFTER:
            last_break = len(cur) - 1
    if cur:
        out.append(prefix + "".join(cur).rstrip())
    return out


def _display_width(text: str) -> int:
    """Return terminal display column width (ANSI-stripped, East Asian Wide/Fullwidth = 2)."""
    plain = _ANSI_RE.sub("", text)
    width = 0
    for ch in plain:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def clean_public_text(val: Optional[str], max_chars: int) -> Optional[str]:
    if not val:
        return None
    s = re.sub(r"\x1b\[[0-9;?]*[A-Za-z~]", "", str(val))
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    return s[:max_chars]

UNIFIED_IDS = {
    "m5ultra", "m5ultra64", "m5max40", "m5max32", "m5pro", "m6", "m5",
    "m3ultra", "m4max", "m4pro", "m4", "m2ultra", "m2max", "aimax395", "dgxspark",
}

HW_ID_LABELS = {
    "rtx5090": "RTX 5090", "rtx5080": "RTX 5080", "rtx5070ti": "RTX 5070 Ti", "rtx5070": "RTX 5070",
    "rtx5060ti16": "RTX 5060 Ti 16GB", "rtx5060": "RTX 5060",
    "rtx4090": "RTX 4090", "rtx4090-48": "RTX 4090 48GB",
    "rtx4080s": "RTX 4080 Super", "rtx4080s-32": "RTX 4080 Super 32GB",
    "rtx4070tis": "RTX 4070 Ti Super", "rtx4070ti": "RTX 4070 Ti", "rtx4070s": "RTX 4070 Super", "rtx4070": "RTX 4070",
    "rtx4060ti16": "RTX 4060 Ti 16GB",
    "rtx3090": "RTX 3090", "rtx3080-20": "RTX 3080 20GB", "rtx3080ti": "RTX 3080 Ti", "rtx3060": "RTX 3060 12GB",
    "rtx2080ti-22": "RTX 2080 Ti 22GB",
    "rtxpro6000": "RTX PRO 6000 Blackwell", "rtxpro5000-72": "RTX PRO 5000 Blackwell 72GB",
    "rtxpro5000": "RTX PRO 5000 Blackwell 48GB", "rtxpro4500": "RTX PRO 4500 Blackwell", "rtxpro4000": "RTX PRO 4000 Blackwell",
    "v100-16": "Tesla V100 16GB", "v100-32": "Tesla V100 32GB",
    "rx7900xtx": "RX 7900 XTX", "rx7900xt": "RX 7900 XT", "r9700": "Radeon AI PRO R9700",
    "rx9070xt": "RX 9070 XT", "rx9070": "RX 9070", "rx9060xt16": "RX 9060 XT 16GB",
    "w7900": "Radeon PRO W7900", "w7800-48": "Radeon PRO W7800 48GB", "mi50": "Instinct MI50 16GB",
    "arcprob70": "Arc Pro B70", "arcprob60": "Arc Pro B60", "arcb580": "Arc B580", "arca770": "Arc A770 16GB",
    "m5ultra": "Apple M5 Ultra (80-core GPU)", "m5ultra64": "Apple M5 Ultra (64-core GPU)",
    "m5max40": "Apple M5 Max (40-core GPU)", "m5max32": "Apple M5 Max (32-core GPU)",
    "m5pro": "Apple M5 Pro", "m6": "Apple M6", "m5": "Apple M5",
    "m3ultra": "Apple M3 Ultra", "m4max": "Apple M4 Max", "m4pro": "Apple M4 Pro", "m4": "Apple M4",
    "m2ultra": "Apple M2 Ultra", "m2max": "Apple M2 Max",
    "aimax395": "AMD Ryzen AI Max+ 395", "dgxspark": "NVIDIA DGX Spark",
}

HW_DEFAULT_VRAM = {
    "rtx5090": 32, "rtx5080": 16, "rtx5070ti": 16, "rtx5070": 12, "rtx5060ti16": 16, "rtx5060": 8,
    "rtx4090": 24, "rtx4090-48": 48, "rtx4080s": 16, "rtx4080s-32": 32,
    "rtx4070tis": 16, "rtx4070ti": 12, "rtx4070s": 12, "rtx4070": 12, "rtx4060ti16": 16,
    "rtx3090": 24, "rtx3080-20": 20, "rtx3080ti": 12, "rtx3060": 12, "rtx2080ti-22": 22,
    "rtxpro6000": 96, "rtxpro5000-72": 72, "rtxpro5000": 48, "rtxpro4500": 32, "rtxpro4000": 24,
    "v100-16": 16, "v100-32": 32,
    "rx7900xtx": 24, "rx7900xt": 20, "r9700": 32, "rx9070xt": 16, "rx9070": 16, "rx9060xt16": 16,
    "w7900": 48, "w7800-48": 48, "mi50": 16,
    "arcprob70": 32, "arcprob60": 24, "arcb580": 12, "arca770": 16,
}


def infer_engine_from_text(text: str) -> Optional[str]:
    low = (text or "").lower()
    # Basalt 是 Strata 的 Blackwell 专用分叉，名字/日志里可能同时出现 strata，必须先判断
    if re.search(r"\bbasalt\b", low):
        return "basalt"
    if re.search(r"\bstrata\b", low):
        return "strata"
    if re.search(r"\b(?:llama\.?cpp|llamacpp|llama[-_]?server|ik[-_]?llama|ollama|lm\s*studio|koboldcpp)\b", low):
        return "llamacpp"
    if re.search(r"\bvllm\b", low):
        return "vllm"
    if re.search(r"\bsglang\b", low):
        return "sglang"
    if re.search(r"\b(?:mlx|omlx|mlx-serve)\b", low):
        return "mlx"
    if re.search(r"\bmaya\b", low):
        return "maya"
    return None


def _detect_local_process_engine(port: Optional[int]) -> Optional[str]:
    """Best-effort engine detection from local listening process name ONLY (no paths/args)."""
    os_name = normalize_os()
    if port:
        if os_name == "linux":
            ss_out = run_cmd(["ss", "-tlnp"], timeout=2.0)
            for line in ss_out.splitlines():
                if f":{port} " in line or f":{port}\t" in line:
                    names = re.findall(r'\(\("([^"]+)"', line)
                    for nm in names:
                        eng = infer_engine_from_text(nm)
                        if eng:
                            return eng
        elif os_name == "macos":
            lsof_out = run_cmd(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=2.0)
            for line in lsof_out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    eng = infer_engine_from_text(parts[0])
                    if eng:
                        return eng
        elif os_name == "windows":
            ps = run_cmd([
                "powershell", "-NoProfile", "-Command",
                f"(Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
                f"Select-Object -First 1 -ExpandProperty OwningProcess) | "
                f"ForEach-Object {{ (Get-Process -Id $_ -ErrorAction SilentlyContinue).ProcessName }}",
            ], timeout=3.0)
            eng = infer_engine_from_text(ps)
            if eng:
                return eng
    return None


def _detect_http_engine(endpoint: str, api_key: Optional[str] = None, shape_guess: bool = True) -> Optional[str]:
    """Inspect local server HTTP Server header or /props build_info for engine name."""
    if not endpoint or not is_local_endpoint(endpoint):
        return None
    ep = endpoint.rstrip("/")
    root = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for url in (f"{root}/props", f"{ep}/models"):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with _urlopen(req, timeout=1.5) as resp:
                srv = resp.headers.get("Server") or resp.headers.get("X-Powered-By") or ""
                eng = infer_engine_from_text(srv)
                if eng:
                    return eng
                if url.endswith("/props"):
                    raw = resp.read().decode("utf-8", errors="ignore")
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        for k in ("build_info", "server", "engine", "version"):
                            v = obj.get(k)
                            if isinstance(v, str):
                                eng = infer_engine_from_text(v)
                                if eng:
                                    return eng
                        model_strs = [obj.get(k) for k in ("model_path", "model_alias", "model")]
                        model_strs = [v for v in model_strs if isinstance(v, str)]
                        # Basalt 的权重是单个 .basalt 文件；GSQ-RCO 量化 Strata 和 Basalt 都支持，不能直接判成 Strata
                        if any(re.search(r"\.basalt\b", v, re.IGNORECASE) for v in model_strs):
                            return "basalt"
                        if any(re.search(r"gsq[\s_-]*rco|ista[\s_-]*daslab", v, re.IGNORECASE) for v in model_strs):
                            return "strata"
                        # 只凭 /props 的结构猜 llama.cpp 是弱信号（Strata/Basalt 也提供同结构的 /props）
                        if shape_guess and any(k in obj for k in ("default_generation_settings", "total_slots")):
                            return "llamacpp"
        except Exception:
            continue
    return None


def infer_engine_auto(endpoint: str, model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    """Best-effort engine detection from model name, local process name, HTTP headers, or port."""
    low_model = (model_name or "").lower()
    if re.search(r"\.basalt\b|\bbasalt\b", low_model):
        return "basalt"
    try:
        port = urllib.parse.urlparse(endpoint).port
    except Exception:
        port = None
    if endpoint and is_local_endpoint(endpoint):
        proc_eng = _detect_local_process_engine(port)
        if proc_eng:
            return proc_eng
        http_eng = _detect_http_engine(endpoint, api_key=api_key, shape_guess=False)
        if http_eng:
            return http_eng
    # 模型名带 GSQ-RCO、又没有进程名 / 服务头 / .basalt 文件等明确线索时，按 Strata 兜底（Basalt 也支持该量化）
    if "gsq-rco" in low_model:
        return "strata"
    if endpoint and is_local_endpoint(endpoint):
        shape_eng = _detect_http_engine(endpoint, api_key=api_key, shape_guess=True)
        if shape_eng:
            return shape_eng
    if port in (11434, 1234):
        return "llamacpp"
    return None


def parse_human_note(text: str) -> Dict[str, Any]:
    """Parse optional natural-language notes: RAM, DDR, channels, OS, engine, quant, MTP."""
    raw = (text or "").strip()
    if not raw:
        return {}
    low = raw.lower()
    out: Dict[str, Any] = {}

    eng = infer_engine_from_text(raw)
    if eng:
        out["engine"] = eng

    q = infer_quant(raw)
    if q:
        out["quant"] = q
    elif re.search(r"\b(?:gsq[\s_-]*rco|gsq|ista[\s_-]*daslab)\b", low):
        out["quant_prefix"] = "GSQ-RCO"
    elif re.search(r"\b(?:unsloth|ud)\b", low):
        out["quant_prefix"] = "UD"

    if re.search(r"(?:不开|关闭|关|no\s*|without\s*)mtp|mtp\s*=\s*(?:false|0|off)", low):
        out["mtp"] = "false"
    elif re.search(r"(?:开|开启)?\bmtp\b|开\s*mtp|mtp\s*=\s*(?:true|1|on)", low):
        out["mtp"] = "true"

    # Extract KV cache precision (e.g. "kv int8", "kv cache q8_0", "fp8 kv", "kv=f16")
    kv_pat = r"(?:f16|fp16|bf16|fp8|f8|int8|q8_0|q8|q4_0|q4|int4|q5_0)(?:/(?:f16|fp16|bf16|fp8|int8|q8_0|q8|q4_0|q4|int4))?"
    m_kv = (
        re.search(rf"(?:kv[\s_-]*(?:cache)?|cache[\s_-]*type(?:[\s_-]*[kv])?)\s*[:=：\s]\s*({kv_pat})\b", low)
        or re.search(rf"\b({kv_pat})\s*kv(?:[\s_-]*cache)?\b", low)
    )
    if m_kv:
        out["kv_cache"] = normalize_kv_cache(m_kv.group(1))

    m_ddr = re.search(r"\b(ddr[345]|lpddr5x?)\b", low)
    if m_ddr:
        out["ram_type"] = m_ddr.group(1).upper()
    elif "x99" in low:
        out["ram_type"] = "DDR4"

    if re.search(r"十二通道|12\s*(?:通道|ch)", low):
        out["ram_channels"] = 12
    elif re.search(r"八通道|8\s*(?:通道|ch)", low):
        out["ram_channels"] = 8
    elif re.search(r"四通道|4\s*(?:通道|ch)|\bx99\b|\bx299\b|quad", low):
        out["ram_channels"] = 4
    elif re.search(r"双通道|2\s*(?:通道|ch)|dual", low):
        out["ram_channels"] = 2

    if re.search(r"\b(?:linux|ubuntu|debian|arch|fedora|centos)\b", low):
        out["os"] = "linux"
    elif re.search(r"\b(?:win|windows|win11|win10)\b", low):
        out["os"] = "windows"
    elif re.search(r"\b(?:mac|macos|osx)\b", low):
        out["os"] = "macos"

    # Extract RAM GB when tied to DDR / RAM / 内存 / '+'
    m_ram = (
        re.search(r"(?:ddr[345]|lpddr5x?|内存|ram)\s*[:：]?\s*(\d{2,4})\s*g(?:b)?\b", low)
        or re.search(r"(\d{2,4})\s*g(?:b)?\s*(?:ddr[345]|lpddr5x?|内存|ram)", low)
        or re.search(r"[+＋]\s*(?:(?:ddr[345]|x99|x299|双通道|四通道|八通道)\s*)*(\d{2,4})\s*g(?:b)?\b", low)
    )
    if m_ram:
        out["ram_gb"] = int(m_ram.group(1))
    return out


def parse_linux_non_nvidia_gpus(
    rocm_lines: List[str],
    rocm_mem_lines: List[str],
    amd_smi_lines: List[str],
    pci_lines: List[str],
    vram_bytes_list: List[int],
) -> List[Tuple[str, Optional[int]]]:
    """Extract (gpu_name, vram_gb) pairs for AMD/Intel GPUs on Linux (local or SSH)."""
    vrams: List[int] = []
    for b in vram_bytes_list:
        if b and b > 0:
            gb = max(1, round(b / (1024 ** 3)))
            vrams.append(gb)
    if not vrams:
        for line in rocm_mem_lines:
            low = line.lower()
            if not line.strip() or low.startswith("device") or "total" in low and not re.search(r"\d{6,}", line):
                continue
            nums = [int(x) for x in re.findall(r"\b(\d{8,})\b", line)]
            if nums:
                vrams.append(max(1, round(nums[0] / (1024 ** 3))))
    # Drop tiny iGPU carveouts (<=2GB) when discrete GPUs (>=6GB) are also present
    if any(v >= 6 for v in vrams):
        vrams = [v for v in vrams if v >= 4]

    def pick_best_name(candidates: List[str], vram: Optional[int]) -> Optional[str]:
        cleaned = []
        for c in candidates:
            s = re.sub(r"\s+", " ", (c or "").strip().strip('"'))
            if not s or s.lower() in ("n/a", "none", "unknown", "card0", "card1", "card2", "card3"):
                continue
            if re.fullmatch(r"0x[0-9a-f]+|\d+", s, re.IGNORECASE):
                continue
            cleaned.append(s)
        for s in cleaned:
            if map_gpu_id(s, vram):
                return s
        for s in cleaned:
            if any(k in s.lower() for k in ("radeon", "instinct", "arc ", "intel arc")):
                return s
        return cleaned[0] if cleaned else None

    gpus: List[Tuple[str, Optional[int]]] = []

    # 1. amd-smi static CSV
    amd_rows = [l for l in amd_smi_lines if l.strip() and not l.lower().startswith(("gpu,", "device"))]
    for idx, line in enumerate(amd_rows):
        cols = [c.strip() for c in line.split(",")]
        vram = vrams[idx] if idx < len(vrams) else None
        if vram is None:
            m_mb = re.search(r"\b(\d{4,6})\s*mb\b", line, re.IGNORECASE)
            if m_mb:
                vram = max(1, round(int(m_mb.group(1)) / 1024))
        name = pick_best_name(cols[1:] if len(cols) > 1 else cols, vram)
        if name:
            gpus.append((name, vram))
    if gpus:
        return gpus

    # 2. rocm-smi --showproductname --csv
    rocm_rows = [l for l in rocm_lines if l.strip() and not l.lower().startswith("device")]
    for idx, line in enumerate(rocm_rows):
        cols = [c.strip() for c in line.split(",")]
        vram = vrams[idx] if idx < len(vrams) else None
        name = pick_best_name(cols[1:] if len(cols) > 1 else cols, vram)
        if name:
            gpus.append((name, vram))

    # 3. lspci fallback (or enrich generic ROCm name when lspci gives a mappable model)
    pci_gpus: List[Tuple[str, Optional[int]]] = []
    skip_pci = ("aspeed", "matrox", "vmware", "qxl", "virtio", "bochs", "raphael", "granite ridge", "phoenix", "rembrandt", "cezanne")
    for line in pci_lines:
        low = line.lower()
        if any(sk in low for sk in skip_pci):
            continue
        if not any(k in low for k in ("amd", "ati", "radeon", "instinct", "navi", "intel", "arc ")):
            continue
        # Strip leading PCI bus + device class prefix
        desc = re.sub(r"^[0-9a-f:.]+\s+[^:]+:\s*", "", line.strip(), flags=re.IGNORECASE)
        desc = re.sub(r"\s*\(rev [^)]+\)\s*$", "", desc).strip()
        # Prefer bracketed product name if present (e.g. "[Radeon AI PRO R9700]")
        brackets = re.findall(r"\[([^\]]+)\]", desc)
        vram = vrams[len(pci_gpus)] if len(pci_gpus) < len(vrams) else None
        cand = pick_best_name(brackets + [desc], vram) or desc
        if cand:
            pci_gpus.append((cand, vram))

    if gpus:
        # If ROCm didn't map cleanly to a known hw_id but lspci + VRAM did, prefer lspci name
        enriched: List[Tuple[str, Optional[int]]] = []
        for idx, (name, vram) in enumerate(gpus):
            if not map_gpu_id(name, vram) and idx < len(pci_gpus):
                p_name, p_vram = pci_gpus[idx]
                use_vram = vram or p_vram
                if map_gpu_id(p_name, use_vram):
                    enriched.append((p_name, use_vram))
                    continue
            enriched.append((name, vram))
        return enriched

    return pci_gpus


def _is_own_interface_ip(host: str) -> bool:
    """Return True if host is an IP address assigned to a local network interface on this machine."""
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, host)
        except OSError:
            continue
        s = socket.socket(family, socket.SOCK_DGRAM)
        try:
            s.bind((host, 0))
            return True
        except OSError:
            return False
        finally:
            s.close()
    return False


def is_local_endpoint(endpoint: str) -> bool:
    """Return True if endpoint points to the local machine (loopback or this host's own interface IP)."""
    try:
        host = (urllib.parse.urlparse(endpoint).hostname or "").lower().strip("[]")
    except Exception:
        return False
    if host in ("localhost", "localhost.localdomain", "::1", "0.0.0.0"):
        return True
    if host.startswith("127."):
        return True
    if host and _is_own_interface_ip(host):
        return True
    return False

COMMON_ENDPOINTS = [
    "http://127.0.0.1:8080/v1",   # llama-server / Strata / mlx-serve
    "http://127.0.0.1:17880/v1",  # Strata desktop / launcher alternate port
    "http://127.0.0.1:9931/v1",   # llama-server newer default
    "http://127.0.0.1:11434/v1",  # Ollama
    "http://127.0.0.1:1234/v1",   # LM Studio
    "http://127.0.0.1:8000/v1",   # vLLM / SGLang
    "http://127.0.0.1:30000/v1",  # SGLang default
    "http://127.0.0.1:8081/v1",   # alternate local port
    "http://127.0.0.1:8082/v1",   # alternate local port
    "http://127.0.0.1:18080/v1",  # alternate Strata / llama-server port
    "http://127.0.0.1:8001/v1",   # alternate vLLM / OpenAI port
    "http://127.0.0.1:9000/v1",   # custom local port
    "http://127.0.0.1:5000/v1",   # alternate local port
]

FILLER_PARAGRAPH = (
    "In local large language model inference, token generation is typically "
    "memory-bandwidth bound during decode and compute bound during prefill. "
    "Mixture-of-Experts architectures reduce active parameters per token while "
    "requiring sufficient VRAM or high-bandwidth system memory to hold expert weights. "
)


def run_cmd(cmd: List[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def normalize_os() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "macos"
    if sys_name.startswith("win"):
        return "windows"
    return "linux"


def map_gpu_id(name: str, vram_gb: Optional[int]) -> Optional[str]:
    """Map raw GPU name + VRAM (GB) to canonical hw_id (supports modded VRAM cards)."""
    s = re.sub(r"\s+", " ", (name or "").lower()).strip()
    v = vram_gb or 0

    # NVIDIA Blackwell PRO / GeForce
    if "rtx pro 6000" in s:
        return "rtxpro6000"
    if "rtx pro 5000" in s:
        return "rtxpro5000-72" if v >= 60 else "rtxpro5000"
    if "rtx pro 4500" in s:
        return "rtxpro4500"
    if "rtx pro 4000" in s:
        return "rtxpro4000"
    if "5090" in s and "laptop" not in s:
        return "rtx5090"
    if "5080" in s and "laptop" not in s:
        return "rtx5080"
    if "5070 ti" in s and "laptop" not in s:
        return "rtx5070ti"
    if "5070" in s and "laptop" not in s:
        return "rtx5070"
    if "5060 ti" in s and v >= 14:
        return "rtx5060ti16"
    if "5060" in s and "ti" not in s and "laptop" not in s:
        return "rtx5060"

    # NVIDIA 40/30/20 series (including modded VRAM variants)
    if "4090" in s and "laptop" not in s:
        return "rtx4090-48" if v >= 40 else "rtx4090"
    if "4080 super" in s:
        return "rtx4080s-32" if v >= 28 else "rtx4080s"
    if "4070 ti super" in s:
        return "rtx4070tis"
    if "4070 ti" in s:
        return "rtx4070ti"
    if "4070 super" in s:
        return "rtx4070s"
    if "4070" in s and "laptop" not in s:
        return "rtx4070"
    if "4060 ti" in s and v >= 14:
        return "rtx4060ti16"
    if "3090" in s:
        return "rtx3090"
    if "3080 ti" in s:
        return "rtx3080ti"
    if "3080" in s and "laptop" not in s and v >= 18:
        return "rtx3080-20"
    if "3060" in s and "ti" not in s and "laptop" not in s and v >= 10:
        return "rtx3060"
    if "2080 ti" in s and v >= 20:
        return "rtx2080ti-22"
    if "v100" in s:
        return "v100-32" if v >= 28 else ("v100-16" if v >= 14 else None)

    # AMD
    if "7900 xtx" in s:
        return "rx7900xtx"
    if "7900 xt" in s:
        return "rx7900xt"
    if "r9700" in s:
        return "r9700"
    if ("9070" in s or "navi 48" in s) and v >= 28:
        return "r9700"
    if "9070 xt" in s:
        return "rx9070xt"
    if "9070" in s:
        return "rx9070"
    if "9060 xt" in s and v >= 14:
        return "rx9060xt16"
    if "w7900" in s:
        return "w7900"
    if "w7800" in s and v >= 40:
        return "w7800-48"
    if "mi50" in s:
        return "mi50"

    # Intel
    if "arc pro b70" in s or "b70" in s and "arc" in s:
        return "arcprob70"
    if "arc pro b60" in s or "b60" in s and "arc" in s:
        return "arcprob60"
    if "arc b580" in s or "b580" in s and "arc" in s:
        return "arcb580"
    if "a770" in s and v >= 14:
        return "arca770"

    return None


def map_apple_id(chip: str, gpu_cores: Optional[int]) -> Optional[str]:
    s = (chip or "").lower()
    c = gpu_cores or 0
    if "m5 ultra" in s:
        return "m5ultra64" if (0 < c <= 68) else "m5ultra"
    if "m5 max" in s:
        return "m5max32" if (0 < c <= 34) else "m5max40"
    if "m5 pro" in s:
        return "m5pro"
    if re.search(r"\bm6\b", s):
        return "m6"
    if re.search(r"\bm5\b", s):
        return "m5"
    if "m4 max" in s:
        return "m4max"
    if "m4 pro" in s:
        return "m4pro"
    if re.search(r"\bm4\b", s):
        return "m4"
    if "m3 ultra" in s:
        return "m3ultra"
    if "m2 ultra" in s:
        return "m2ultra"
    if "m2 max" in s:
        return "m2max"
    return None


KNOWN_LEADERBOARD_MODELS: List[Tuple[str, str]] = [
    ("qwen38-flash-next", "Qwen3.8-Flash-Next"),
    ("qwen38-27b", "Qwen3.8-27B"),
    ("qwen36-35b-a3b", "Qwen3.6-35B-A3B"),
    ("qwen36-27b", "Qwen3.6-27B"),
    ("qwen35-9b", "Qwen3.5-9B"),
    ("qwen35-4b", "Qwen3.5-4B"),
    ("ds-v41-flash", "DeepSeek-V4.1-Flash"),
    ("glm-53-flash", "GLM-5.3-Flash"),
    ("kimi-k3", "Kimi-K3"),
    ("gpt-oss-120b", "gpt-oss-120b"),
    ("gpt-oss-20b", "gpt-oss-20b"),
    ("gemma4-31b", "Gemma-4-31B"),
    ("gemma4-26b-a4b", "Gemma-4-26B-A4B"),
    ("gemma4-12b", "Gemma-4-12B"),
]


def map_model_id(raw: str) -> Optional[str]:
    s = (raw or "").lower()
    if "kimi" in s and "k3" in s:
        return "kimi-k3"
    if "deepseek" in s and ("4.1" in s or "41" in s) and "flash" in s:
        return "ds-v41-flash"
    if "glm" in s and ("5.3" in s or "53" in s) and "flash" in s:
        return "glm-53-flash"
    if "qwen" in s and ("3.8" in s or "38" in s) and "flash" in s and "next" in s:
        return "qwen38-flash-next"
    if "qwen" in s and ("3.8" in s or "38" in s) and "27b" in s:
        return "qwen38-27b"
    if "qwen" in s and ("3.6" in s or "36" in s) and "27b" in s:
        return "qwen36-27b"
    if "qwen" in s and ("3.6" in s or "36" in s) and "35b" in s:
        return "qwen36-35b-a3b"
    if "qwen" in s and ("3.5" in s or "35" in s) and "9b" in s:
        return "qwen35-9b"
    if "qwen" in s and ("3.5" in s or "35" in s) and "4b" in s:
        return "qwen35-4b"
    if "gpt-oss" in s and "120b" in s:
        return "gpt-oss-120b"
    if "gpt-oss" in s and "20b" in s:
        return "gpt-oss-20b"
    if "gemma" in s and "4" in s and "31b" in s:
        return "gemma4-31b"
    if "gemma" in s and "4" in s and "26b" in s:
        return "gemma4-26b-a4b"
    if "gemma" in s and "4" in s and "12b" in s:
        return "gemma4-12b"
    return None


def infer_quant(raw: str) -> Optional[str]:
    s = raw or ""
    has_gsq = bool(re.search(r"gsq[\s_-]*rco|ista[\s_-]*daslab", s, re.IGNORECASE))
    m = re.search(
        r"(UD-(?:IQ[1-4]_[A-Z]+|Q[2-8]_K_[A-Z]+)"
        r"|IQ[1-4]_(?:XXS|XS|S|M|NL|KS|K)"
        r"|Q[2-8]_K(?:_(?:XL|S|M|L))?"
        r"|Q[2-8]_[01]"
        r"|MXFP4|NVFP4|W4A16|W8A8|FP8|BF16|AWQ|GPTQ"
        r"|[2-8]-?bit)",
        s,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).strip()
    norm = val.lower() if val.lower().endswith("bit") else val.upper()
    if has_gsq and not norm.startswith("GSQ-RCO"):
        base = norm[3:] if norm.startswith("UD-") else norm
        return f"GSQ-RCO {base}"
    return norm


STANDARD_RAM_GB = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def snap_ram_gb(raw_gb: int) -> int:
    """Snap OS-reported usable RAM (e.g. 61GB due to kernel/iGPU carveout) to physical installed RAM (64GB)."""
    if raw_gb <= 0:
        return raw_gb
    for std in STANDARD_RAM_GB:
        if std * 0.90 <= raw_gb <= std:
            return std
    return raw_gb


def detect_ram_gb() -> Optional[int]:
    sys_name = normalize_os()
    if sys_name == "macos":
        out = run_cmd(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return snap_ram_gb(max(1, round(int(out) / (1024 ** 3))))
    elif sys_name == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return snap_ram_gb(max(1, round(kb / (1024 ** 2))))
        except Exception:
            pass
    elif sys_name == "windows":
        out = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if out.isdigit():
            return snap_ram_gb(max(1, round(int(out) / (1024 ** 3))))
    return None


def clean_cpu_name(raw: Optional[str]) -> Optional[str]:
    """Normalize verbose OS CPU brand strings into clean, readable model names."""
    if not raw:
        return None
    s = re.sub(r"\((?:R|TM|C)\)", "", str(raw), flags=re.IGNORECASE)
    s = re.sub(r"^\d+(?:st|nd|rd|th)\s+Gen\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+w(?:ith|/)?\s+Radeon(?:\(TM\))?\s+[\w\s-]*Graphics\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\d+-Core\s+Processor\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(?:Dual|Quad|Six|Eight|Twelve|Sixteen)-Core\s+Processor\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+CPU\s*@\s*[\d.]+\s*[GM]Hz\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*@\s*[\d.]+\s*[GM]Hz\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+Processor\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s[:80] if s else None


def detect_cpu_raw(os_name: Optional[str] = None) -> Optional[str]:
    """Detect local CPU model name without requiring elevated privileges."""
    sys_name = os_name or normalize_os()
    raw = ""
    if sys_name == "macos":
        raw = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys_name == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        raw = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        if not raw:
            lscpu = run_cmd(["lscpu"])
            for line in lscpu.splitlines():
                if line.lower().startswith("model name:"):
                    raw = line.split(":", 1)[1].strip()
                    break
    elif sys_name == "windows":
        raw = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ])
    return clean_cpu_name(raw)


def parse_smbios_memory_type(
    smbios_code: Optional[int],
    mem_type: Optional[int] = None,
    speed_mhz: Optional[int] = None,
) -> Optional[str]:
    """Map DMTF SMBIOS MemoryType code (with conservative speed fallback) to DDR generation."""
    code_map = {
        24: "DDR3",
        26: "DDR4",
        27: "LPDDR3",
        28: "LPDDR3",
        29: "LPDDR3",
        30: "LPDDR4",
        34: "DDR5",
        35: "LPDDR5",
        36: "LPDDR5X",
    }
    if smbios_code in code_map:
        return code_map[smbios_code]
    if mem_type == 24:
        return "DDR3"
    if speed_mhz and speed_mhz > 0:
        if speed_mhz >= 4800:
            return "DDR5"
        if 2133 <= speed_mhz <= 3600:
            return "DDR4"
        if 800 <= speed_mhz <= 1866:
            return "DDR3"
    return None


def infer_ram_type_from_cpu(cpu_raw: Optional[str]) -> Optional[str]:
    """High-confidence DDR4/DDR5 inference from CPU platform (leaves dual-support CPUs as None)."""
    if not cpu_raw:
        return None
    s = cpu_raw.lower()

    # Explicitly leave dual-support platforms (Intel 12th/13th/14th Gen) as None
    if re.search(r"\bi[3579]-1[234]\d{3}[a-z0-9]*\b", s):
        return None

    # DDR5-only platforms
    if (
        re.search(r"ryzen\s+[3579]\s+[789]\d{3}[a-z0-9]*\b", s)
        or re.search(r"ryzen\s+ai\s+", s)
        or re.search(r"threadripper.*\b[79]\d{3}[a-z0-9]*\b", s)
        or re.search(r"epyc\s+[89]\d{2}[45][a-z0-9]*\b", s)
        or re.search(r"core\s+ultra\b", s)
        or re.search(r"xeon.*?\b(?:84|64|54|44|34|85|65|55|45|67|69)\d{2}[a-z0-9]*\b", s)
    ):
        return "DDR5"

    # DDR4-only platforms
    if (
        re.search(r"ryzen\s+[3579]\s+[12345]\d{3}[a-z0-9]*\b", s)
        or re.search(r"threadripper.*\b[1235]\d{3}[a-z0-9]*\b", s)
        or re.search(r"epyc\s+7\d{2}[123][a-z0-9]*\b", s)
        or re.search(r"\bi[3579]-(?:[6789]\d{3}|10\d{3}|11\d{3})[a-z0-9]*\b", s)
        or re.search(r"e5-\d{4}\s*v[34]\b", s)
        or re.search(r"xeon.*?\b(?:81|61|51|41|31|82|62|52|42|32|83|63|53|43)\d{2}[a-z0-9]*\b", s)
    ):
        return "DDR4"

    return None


def _parse_dmidecode_ram_type(text: str) -> Optional[str]:
    """Extract DDR generation from `dmidecode -t memory` output."""
    for line in text.splitlines():
        m = re.match(r"^\s*Type:\s*(LPDDR5X|LPDDR5|LPDDR4X|LPDDR4|DDR5|DDR4|DDR3)\b", line, re.IGNORECASE)
        if m:
            val = m.group(1).upper()
            return "LPDDR4" if val == "LPDDR4X" else val
    return None


def detect_ram_type(os_name: Optional[str] = None, cpu_raw: Optional[str] = None) -> Optional[str]:
    """Detect system RAM generation (DDR4/DDR5/LPDDR5/DDR3) on Windows/Linux with CPU fallback."""
    sys_name = os_name or normalize_os()
    if sys_name == "windows":
        ps = run_cmd([
            "powershell", "-NoProfile", "-Command",
            (
                "Get-CimInstance Win32_PhysicalMemory -ErrorAction SilentlyContinue | "
                "ForEach-Object { \"$($_.SMBIOSMemoryType)|$($_.MemoryType)|$($_.ConfiguredClockSpeed)|$($_.Speed)\" }"
            ),
        ])
        for line in ps.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            smbios = int(parts[0]) if parts[0].isdigit() else None
            mtype = int(parts[1]) if parts[1].isdigit() else None
            spd = int(parts[2]) if parts[2].isdigit() and int(parts[2]) > 0 else (
                int(parts[3]) if parts[3].isdigit() and int(parts[3]) > 0 else None
            )
            parsed = parse_smbios_memory_type(smbios, mtype, spd)
            if parsed:
                return parsed
    elif sys_name == "linux":
        # 1. EDAC sysfs (often readable by non-root users on AMD/Intel Linux)
        try:
            edac_root = "/sys/devices/system/edac/mc"
            if os.path.isdir(edac_root):
                for mc in sorted(os.listdir(edac_root)):
                    mc_dir = os.path.join(edac_root, mc)
                    if not os.path.isdir(mc_dir):
                        continue
                    for entry in sorted(os.listdir(mc_dir)):
                        p = os.path.join(mc_dir, entry, "mem_type")
                        if os.path.isfile(p):
                            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                                txt = f.read().strip().upper()
                            m = re.search(r"(LPDDR5X|LPDDR5|LPDDR4|DDR5|DDR4|DDR3)", txt)
                            if m:
                                return m.group(1)
        except Exception:
            pass
        # 2. dmidecode (works when root or CAP_SYS_RAWIO)
        dmi_out = run_cmd(["dmidecode", "-t", "memory"], timeout=3.0)
        if dmi_out:
            parsed = _parse_dmidecode_ram_type(dmi_out)
            if parsed:
                return parsed

    return infer_ram_type_from_cpu(cpu_raw)


def detect_hardware() -> Dict[str, Any]:
    """Collect hardware specs ONLY (no hostname, username, UUID, serial, or paths)."""
    LAST_DETECT_NOTES.clear()
    os_name = normalize_os()
    ram_gb = detect_ram_gb()
    cpu_raw = detect_cpu_raw(os_name)

    # 1. Apple Silicon (unified memory)
    if os_name == "macos" and platform.machine() == "arm64":
        chip = cpu_raw or "Apple Silicon"
        disp = run_cmd(["system_profiler", "SPDisplaysDataType"])
        cores_match = re.search(r"Total Number of Cores:\s*(\d+)", disp)
        gpu_cores = int(cores_match.group(1)) if cores_match else None
        hw_raw = f"{chip} ({gpu_cores}-core GPU)" if gpu_cores else chip
        return {
            "kind": "unified",
            "hw_id": map_apple_id(chip, gpu_cores),
            "hw_raw": hw_raw,
            "cpu_raw": chip,
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": ram_gb,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": os_name,
        }

    # 2. 收集各厂商显卡。NVIDIA 额外读取已用显存，用来判断模型实际加载在哪些卡上
    nv = parse_nvidia_smi(run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]))
    if nv and "gb10" in nv[0]["name"].lower():
        # DGX Spark (GB10 unified memory)
        return {
            "kind": "unified",
            "hw_id": "dgxspark",
            "hw_raw": nv[0]["name"],
            "cpu_raw": cpu_raw,
            "gpu_count": len(nv),
            "vram_gb": None,
            "ram_gb": ram_gb or 128,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": os_name,
        }

    def _gpu_spec(name: str, vram: Optional[int], count: int) -> Dict[str, Any]:
        return {
            "kind": "gpu",
            "hw_id": map_gpu_id(name, vram),
            "hw_raw": name,
            "cpu_raw": cpu_raw,
            "gpu_count": count,
            "vram_gb": vram or None,
            "ram_gb": ram_gb,
            "ram_type": detect_ram_type(os_name, cpu_raw),
            "ram_channels": None,
            "os": os_name,
        }

    other = detect_non_nvidia_gpus(os_name)
    other_dgpu = [g for g in other if map_gpu_id(g[0], g[1]) or (g[1] or 0) >= 8]

    nv_active = [g for g in nv if g["used_mb"] is not None and g["used_mb"] >= NV_ACTIVE_USED_MB]
    nv_known_idle = bool(nv) and all(g["used_mb"] is not None and g["used_mb"] < NV_ACTIVE_USED_MB for g in nv)

    if nv and not (nv_known_idle and other_dgpu):
        pool = nv_active or nv
        name, vram, count, others = choose_gpu_group([(g["name"], g["vram_gb"]) for g in pool])
        if nv_active and len(nv_active) < len(nv):
            _note(f"检测到 {len(nv)} 张 NVIDIA 显卡，其中 {len(nv_active)} 张已加载模型（显存占用 ≥{NV_ACTIVE_USED_MB // 1024}GB），按这 {len(nv_active)} 张中的同型号卡统计。")
        mix = gpu_mix_from(pool) if (nv_active and others) else None
        if mix:
            _note(f"检测到多卡混搭且都已加载模型（{format_gpu_mix(mix)}），成绩会标记为混搭配置，不按单卡 {name} 排名。")
        elif others:
            _note(f"检测到不同型号的显卡（{'、'.join(others)}），按 {name} ×{count} 统计；如不对请用 --hw-id / --gpu-count 手动指定。")
        if other_dgpu:
            _note(f"同时检测到非 NVIDIA 显卡（{'、'.join(g[0] for g in other_dgpu)}），按 NVIDIA 显卡统计；如模型实际跑在其它卡上请用 --hw-id 指定。")
        spec = _gpu_spec(name, vram, count)
        if mix:
            spec["gpu_mix"] = mix
        return spec

    if nv and other_dgpu:
        _note(f"NVIDIA 显卡（{nv[0]['name']}）上未发现已加载的模型，按 {other_dgpu[0][0]} 统计；如不对请用 --hw-id 手动指定。")

    # 3. AMD Ryzen AI Max+ 395 (unified) on Linux/Windows
    if cpu_raw and "ai max" in cpu_raw.lower() and "395" in cpu_raw.lower() and not other_dgpu:
        return {
            "kind": "unified",
            "hw_id": "aimax395",
            "hw_raw": cpu_raw,
            "cpu_raw": cpu_raw,
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": ram_gb,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": os_name,
        }

    # 4. AMD / Intel GPU
    if other:
        name, vram, count, others = choose_gpu_group(other_dgpu or other)
        if others:
            _note(f"检测到不同型号的显卡（{'、'.join(others)}），按 {name} ×{count} 统计；如不对请用 --hw-id / --gpu-count 手动指定。")
        return _gpu_spec(name, vram, count)

    return {
        "kind": "gpu",
        "hw_id": None,
        "hw_raw": cpu_raw or "Unknown Hardware",
        "cpu_raw": cpu_raw,
        "gpu_count": 1,
        "vram_gb": None,
        "ram_gb": ram_gb,
        "ram_type": detect_ram_type(os_name, cpu_raw),
        "ram_channels": None,
        "os": os_name,
    }


# 显存占用达到该值的 NVIDIA 卡视为「已加载模型」（桌面/浏览器通常只占几百 MB）
NV_ACTIVE_USED_MB = 2048
# 最近一次硬件识别的提示（不会上传；CLI 打印、WebUI 写入日志）
LAST_DETECT_NOTES: List[str] = []


def _note(msg: str) -> None:
    if msg not in LAST_DETECT_NOTES:
        LAST_DETECT_NOTES.append(msg)


def parse_nvidia_smi(out: str) -> List[Dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits`."""
    gpus: List[Dict[str, Any]] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            vram = max(1, round(float(parts[1]) / 1024))
        except ValueError:
            vram = 0
        used: Optional[int] = None
        if len(parts) >= 3:
            try:
                used = int(float(parts[2]))
            except ValueError:
                used = None  # e.g. "[N/A]"
        gpus.append({"name": parts[0], "vram_gb": vram, "used_mb": used})
    return gpus


def gpu_mix_from(gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group GPUs holding the model by model name: [{"name", "vram_gb", "count"}], largest first."""
    mix: List[Dict[str, Any]] = []
    for g in gpus:
        for m in mix:
            if m["name"] == g["name"]:
                m["count"] += 1
                break
        else:
            mix.append({"name": g["name"], "vram_gb": g.get("vram_gb"), "count": 1})
    for m in mix:
        # 服务端按各卡 hw_id 查带宽，核验混搭配置的物理速度上限
        m["hw_id"] = map_gpu_id(m["name"], m["vram_gb"])
    mix.sort(key=lambda m: (1 if m["hw_id"] else 0, m["vram_gb"] or 0), reverse=True)
    return mix


def format_gpu_mix(mix: List[Dict[str, Any]]) -> str:
    short = lambda n: re.sub(r"^(?:NVIDIA\s+)?(?:GeForce\s+)?", "", n).strip()
    return " + ".join(f"{short(m['name'])}" + (f" ×{m['count']}" if m["count"] > 1 else "") for m in mix)


def choose_gpu_group(cands: List[Tuple[str, Optional[int]]]) -> Tuple[str, Optional[int], int, List[str]]:
    """Pick the primary GPU model and count only cards of that same model.

    Mixed setups (e.g. 4090 + 3060) must not be reported as "4090 ×2". Prefers a card that maps
    to a known hw_id, then the largest VRAM. Returns (name, vram_gb, count, other_model_names).
    """
    order = sorted(
        range(len(cands)),
        key=lambda i: (1 if map_gpu_id(cands[i][0], cands[i][1]) else 0, cands[i][1] or 0, -i),
        reverse=True,
    )
    name, vram = cands[order[0]]
    count = sum(1 for c in cands if c[0] == name)
    others: List[str] = []
    for c in cands:
        if c[0] != name and c[0] not in others:
            others.append(c[0])
    return name, vram, count, others


# Windows：以当前「在位」的显示适配器（Win32_VideoController）为准，注册表只用来查 64 位显存大小。
# 直接枚举注册表会带出已拔掉/旧驱动残留的显卡记录，导致张数或型号识别错误。
_WIN_GPU_PS = (
    "$reg = Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
    "-ErrorAction SilentlyContinue | Where-Object { $_.DriverDesc }; "
    "Get-CimInstance Win32_VideoController | Where-Object { $_.Name } | ForEach-Object { "
    "$v = $_; $n = $v.Name; "
    "$m = $reg | Where-Object { $_.DriverDesc -eq $n } | ForEach-Object { $_.'HardwareInformation.qwMemorySize' } "
    "| Where-Object { $_ } | Sort-Object -Descending | Select-Object -First 1; "
    "if (-not $m) { $m = $v.AdapterRAM }; \"$n|$m\" }"
)
_WIN_SKIP = ("microsoft basic", "microsoft remote", "virtual", "parsec", "citrix", "vmware", "nvidia")


def parse_windows_gpu_lines(text: str) -> List[Tuple[str, Optional[int]]]:
    """Parse 'name|bytes' lines from _WIN_GPU_PS into non-NVIDIA (name, vram_gb) pairs."""
    cands: List[Tuple[str, Optional[int]]] = []
    for line in (text or "").splitlines():
        if "|" not in line:
            continue
        name, ram_b = line.split("|", 1)
        name = name.strip()
        ram_b = ram_b.strip()
        if not name or any(sk in name.lower() for sk in _WIN_SKIP):
            continue
        vram = max(1, round(int(ram_b) / (1024 ** 3))) if ram_b.isdigit() and int(ram_b) > 0 else None
        cands.append((name, vram))
    return cands


def detect_non_nvidia_gpus(os_name: str) -> List[Tuple[str, Optional[int]]]:
    """AMD / Intel GPUs on Windows or Linux as (name, vram_gb) pairs."""
    if os_name == "windows":
        return parse_windows_gpu_lines(run_cmd(["powershell", "-NoProfile", "-Command", _WIN_GPU_PS]))
    if os_name == "linux":
        amd_smi = run_cmd(["amd-smi", "static", "--asic", "--vram", "--csv"]).splitlines()
        rocm = run_cmd(["rocm-smi", "--showproductname", "--csv"]).splitlines()
        rocm_mem = run_cmd(["rocm-smi", "--showmeminfo", "vram", "--csv"]).splitlines()
        pci = [
            l for l in run_cmd(["lspci"]).splitlines()
            if re.search(r"VGA|3D|Display", l, re.IGNORECASE) and "nvidia" not in l.lower()
        ]
        vram_bytes: List[int] = []
        try:
            drm_dir = "/sys/class/drm"
            if os.path.isdir(drm_dir):
                for entry in sorted(os.listdir(drm_dir)):
                    if not re.fullmatch(r"card\d+", entry):
                        continue
                    vp = os.path.join(drm_dir, entry, "device", "mem_info_vram_total")
                    if os.path.isfile(vp):
                        with open(vp, "r", encoding="utf-8") as f:
                            raw_b = f.read().strip()
                            if raw_b.isdigit():
                                vram_bytes.append(int(raw_b))
        except Exception:
            pass
        return parse_linux_non_nvidia_gpus(rocm, rocm_mem, amd_smi, pci, vram_bytes)
    return []


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(req: urllib.request.Request, timeout: float = 15.0) -> Any:
    """Bypass system HTTP/SOCKS proxies for local endpoints while keeping proxy for external uploads."""
    if is_local_endpoint(req.full_url):
        return _NO_PROXY_OPENER.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _extract_model_ids(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    raw_list = data.get("data") if isinstance(data.get("data"), list) else (
        data.get("models") if isinstance(data.get("models"), list) else []
    )
    out: List[str] = []
    for m in raw_list:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model") or m.get("name")
            if mid and str(mid) not in out:
                out.append(str(mid))
    return out


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Any:
    hdrs = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data is not None else "GET")
    with _urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


COMMON_PORTS = [8080, 17880, 9931, 11434, 1234, 8000, 30000, 8081, 8082, 18080, 8001, 9000, 5000]


def endpoint_candidates(raw: str) -> List[str]:
    """Generate ordered candidate local base URLs from flexible user input.

    Accepts:
      - '8080' or ':8080'                      (expands to http://127.0.0.1:8080/v1 & root)
      - '127.0.0.1:11434' / 'localhost:8080'   (auto-adds http:// and tries /v1 & root)
      - 'http://127.0.0.1:8080/v1'             (uses as-is)
      - 'http://127.0.0.1:8080/v1/chat/completions' (strips /chat/completions)
    """
    s = (raw or "").strip()
    if not s:
        return []
    m_port = re.fullmatch(r":?(\d{2,5})", s)
    if m_port:
        s = f"http://127.0.0.1:{m_port.group(1)}"
    elif not re.match(r"^https?://", s, re.IGNORECASE):
        s = "http://" + s
    s = s.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break

    parsed = urllib.parse.urlparse(s)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or ""
    if not host:
        return [s]
    host_fmt = f"[{host}]" if ":" in host else host
    port = parsed.port
    path = (parsed.path or "").rstrip("/")

    cands: List[str] = []

    def add(url: str) -> None:
        u = url.rstrip("/")
        if u not in cands:
            cands.append(u)

    if port is not None:
        base = f"{scheme}://{host_fmt}:{port}"
        if path:
            add(f"{base}{path}")
            if not path.lower().endswith("/v1"):
                add(f"{base}{path}/v1")
        else:
            add(f"{base}/v1")
            add(base)
    else:
        base = f"{scheme}://{host_fmt}"
        if path:
            add(f"{base}{path}")
            if not path.lower().endswith("/v1"):
                add(f"{base}{path}/v1")
        else:
            add(f"{base}/v1")
            add(base)
            for p in COMMON_PORTS:
                add(f"{scheme}://{host_fmt}:{p}/v1")
                add(f"{scheme}://{host_fmt}:{p}")
    return cands


LOCAL_ONLY_ERR = (
    "为保证硬件规格真实可信，本工具仅支持在运行大模型的本机进行测速（仅允许 127.0.0.1 / localhost）。\n"
    "如推理服务运行在另一台电脑或服务器上，请直接到那台机器上（或通过 SSH 登录后）运行本脚本。"
)


class AuthRequiredError(RuntimeError):
    """Raised when a local inference server is found but requires an API key (HTTP 401/403)."""
    def __init__(self, endpoint: str, status_code: int = 401) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(
            f"已发现本机服务 {endpoint}，但该服务开启了 API Key 验证（HTTP {status_code}）。请传入 --api-key <密钥>。"
        )


def resolve_endpoint(raw: str, api_key: Optional[str] = None, explicit_model: Optional[str] = None) -> Tuple[str, List[str]]:
    """Normalize and probe a user-supplied local endpoint (rejects non-local hosts)."""
    cands = endpoint_candidates(raw)
    if not cands:
        raise RuntimeError("接口地址不能为空。")
    if not is_local_endpoint(cands[0]):
        raise ValueError(LOCAL_ONLY_ERR)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err: Optional[Exception] = None
    auth_cand: Optional[Tuple[str, int]] = None
    for cand in cands:
        try:
            data = http_json(f"{cand}/models", headers=headers, timeout=2.5)
            models = _extract_model_ids(data)
            if models:
                return cand, [explicit_model] if explicit_model else models
            if explicit_model:
                return cand, [explicit_model]
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and auth_cand is None:
                auth_cand = (cand, e.code)
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    if auth_cand:
        raise AuthRequiredError(auth_cand[0], auth_cand[1])
    if explicit_model:
        return cands[0], [explicit_model]
    raise RuntimeError(
        f"连不上本机接口 {raw}（已尝试 {', '.join(cands[:3])}{' 等' if len(cands) > 3 else ''}）。"
        f"请确认服务已启动并加载模型（{last_err}）。"
    )


_LLM_PROC_TIER1 = (
    "strata", "basalt", "llama", "ollama", "lmstudio", "lm-studio", "lms",
    "vllm", "sglang", "mlx", "maya", "kobold", "tabby", "jan",
    "gpt4all", "localai", "exllama", "ik_llama", "llamafile",
)
_LLM_PROC_TIER2 = (
    "python", "python3", "pythonw", "uvicorn", "gunicorn",
    "node", "bun", "deno", "electron",
)
_SKIP_LISTEN_PROCS = (
    "svchost", "system", "lsass", "services", "wininit", "spoolsv",
    "searchhost", "msmpeng", "dashost", "winlogon", "csrss", "smss",
    "chrome", "msedge", "firefox", "safari", "brave", "opera",
    "clash", "v2ray", "mihomo", "sing-box", "shadowsocks", "xray",
    "wechat", "qq", "discord", "slack", "spotify", "steam", "epicgameslauncher",
    "mysqld", "postgres", "redis-server", "mongod", "nginx", "httpd", "sshd",
    "cupsd", "controlcenter", "rapportd", "airplayxpchelper", "mDNSResponder".lower(),
)


def _rank_listening_process(proc_name: str, is_loopback_only: bool = False) -> int:
    """Return priority rank (1=known LLM, 2=runtime, 3=other loopback, 4=other local, 0=skip)."""
    low = (proc_name or "").strip().lower()
    if low.endswith(".exe"):
        low = low[:-4]
    if low:
        if any(k in low for k in _LLM_PROC_TIER1):
            return 1
        if any(low == k or low.startswith(k) for k in _LLM_PROC_TIER2):
            return 2
        if any(low == k or low.startswith(k) for k in _SKIP_LISTEN_PROCS):
            return 0
        return 3 if is_loopback_only else 4
    return 3 if is_loopback_only else 0


def _is_loopback_bind(addr: str) -> bool:
    a = (addr or "").strip().strip("[]").split("%")[0].lower()
    return a == "::1" or a.startswith("127.")


def _normalize_listen_hosts(addr: str) -> List[str]:
    """Map a listening bind address to loopback probe host(s)."""
    a = (addr or "").strip().strip("[]").split("%")[0].lower()
    if a in ("", "0.0.0.0", "*"):
        return ["127.0.0.1"]
    if a.startswith("127."):
        return [a]
    if a == "::":
        return ["127.0.0.1", "[::1]"]
    if a == "::1":
        return ["[::1]", "127.0.0.1"]
    if _is_own_interface_ip(a):
        return [f"[{a}]" if ":" in a else a]
    return []


def _parse_windows_tasklist(csv_output: str) -> Dict[str, str]:
    """Parse 'tasklist /FO CSV /NH' into {pid_str: process_name}."""
    pid_map: Dict[str, str] = {}
    for row in csv.reader((csv_output or "").splitlines()):
        if len(row) >= 2:
            name = row[0].strip()
            pid = row[1].strip()
            if pid.isdigit() and name:
                pid_map[pid] = name
    return pid_map


def _parse_windows_netstat(netstat_output: str, pid_map: Optional[Dict[str, str]] = None) -> List[Tuple[int, str, int]]:
    """Parse 'netstat -ano' TCP LISTENING rows into (rank, host, port)."""
    pm = pid_map or {}
    found: List[Tuple[int, str, int]] = []
    for line in (netstat_output or "").splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        if parts[3].upper() != "LISTENING":
            continue
        m = re.match(r"^(.+):(\d+)$", parts[1])
        if not m:
            continue
        addr, port = m.group(1), int(m.group(2))
        if port < 1024 or port > 65535:
            continue
        pid = parts[4].strip()
        proc = pm.get(pid, "")
        rank = _rank_listening_process(proc, is_loopback_only=_is_loopback_bind(addr))
        if not rank:
            continue
        for h in _normalize_listen_hosts(addr):
            found.append((rank, h, port))
    return found


def _parse_windows_listening(ps_output: str) -> List[Tuple[int, str, int]]:
    """Parse 'LocalAddress|LocalPort|ProcessName' lines from PowerShell."""
    found: List[Tuple[int, str, int]] = []
    for line in (ps_output or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        addr, port_s, proc = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not port_s.isdigit():
            continue
        port = int(port_s)
        if port < 1024 or port > 65535:
            continue
        rank = _rank_listening_process(proc, is_loopback_only=_is_loopback_bind(addr))
        if not rank:
            continue
        for h in _normalize_listen_hosts(addr):
            found.append((rank, h, port))
    return found


def _parse_macos_listening(lsof_output: str) -> List[Tuple[int, str, int]]:
    """Parse 'lsof -nP -iTCP -sTCP:LISTEN' output."""
    found: List[Tuple[int, str, int]] = []
    for line in (lsof_output or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        proc = parts[0]
        addr_port = parts[-2] if parts[-1] == "(LISTEN)" else parts[-1]
        m = re.match(r"^(.+):(\d+)$", addr_port)
        if not m:
            continue
        addr, port = m.group(1), int(m.group(2))
        if port < 1024 or port > 65535:
            continue
        rank = _rank_listening_process(proc, is_loopback_only=_is_loopback_bind(addr))
        if not rank:
            continue
        for h in _normalize_listen_hosts(addr):
            found.append((rank, h, port))
    return found


def _parse_linux_listening(ss_output: str) -> List[Tuple[int, str, int]]:
    """Parse 'ss -tlnp' output."""
    found: List[Tuple[int, str, int]] = []
    for line in (ss_output or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        m = re.match(r"^(.+):(\d+)$", parts[3])
        if not m:
            continue
        addr, port = m.group(1), int(m.group(2))
        if port < 1024 or port > 65535:
            continue
        is_lb = _is_loopback_bind(addr)
        names = re.findall(r'\(\("([^"]+)"', line)
        best_rank = 0
        if names:
            for nm in names:
                r = _rank_listening_process(nm, is_loopback_only=is_lb)
                if r and (best_rank == 0 or r < best_rank):
                    best_rank = r
        else:
            best_rank = _rank_listening_process("", is_loopback_only=is_lb)
        if not best_rank:
            continue
        for h in _normalize_listen_hosts(addr):
            found.append((best_rank, h, port))
    return found


def _discover_listening_endpoints(exclude_urls: Optional[set] = None, max_endpoints: int = 40) -> List[str]:
    """Best-effort discovery of local listening ports (process name only, fast native tools)."""
    os_name = normalize_os()
    entries: List[Tuple[int, str, int]] = []
    if os_name == "windows":
        ns_out = run_cmd(["netstat", "-ano"], timeout=3.0)
        tl_out = run_cmd(["tasklist", "/FO", "CSV", "/NH"], timeout=3.0)
        entries = _parse_windows_netstat(ns_out, _parse_windows_tasklist(tl_out))
    elif os_name == "macos":
        lsof_out = run_cmd(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=2.5)
        entries = _parse_macos_listening(lsof_out)
    elif os_name == "linux":
        ss_out = run_cmd(["ss", "-tlnp"], timeout=2.5)
        entries = _parse_linux_listening(ss_out)

    entries.sort(key=lambda x: (x[0], x[2]))
    seen = set(exclude_urls or ())
    out: List[str] = []
    for _rank, host, port in entries:
        url = f"http://{host}:{port}/v1"
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_endpoints:
            break
    return out


def _is_empty_models_payload(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("data"), list) and len(data["data"]) == 0:
        return True
    if isinstance(data.get("models"), list) and len(data["models"]) == 0:
        return True
    return False


def discover_endpoint(api_key: Optional[str] = None) -> Tuple[str, List[str]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    auth_cand: Optional[Tuple[str, int]] = None
    empty_cand: Optional[str] = None

    def _probe_one(ep: str, timeout: float) -> Tuple[Optional[Tuple[str, List[str]]], Optional[Tuple[str, int]], Optional[str]]:
        ep_clean = ep.rstrip("/")
        root_clean = ep_clean.removesuffix("/v1") if hasattr(str, "removesuffix") else re.sub(r"/v1$", "", ep_clean)
        local_auth: Optional[Tuple[str, int]] = None
        local_empty: Optional[str] = None
        for cand in (ep_clean, root_clean):
            try:
                data = http_json(f"{cand}/models", headers=headers, timeout=timeout)
                models = _extract_model_ids(data)
                if models:
                    return (cand, models), local_auth, local_empty
                if local_empty is None and _is_empty_models_payload(data):
                    local_empty = cand
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and local_auth is None:
                    local_auth = (cand, e.code)
                continue
            except Exception:
                continue
        return None, local_auth, local_empty

    def _probe_list(endpoints: List[str], timeout: float) -> Optional[Tuple[str, List[str]]]:
        nonlocal auth_cand, empty_cand
        if not endpoints:
            return None
        workers = min(16, len(endpoints))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_probe_one, ep, timeout) for ep in endpoints]
            for fut in futures:
                hit, l_auth, l_empty = fut.result()
                if hit:
                    return hit
                if auth_cand is None and l_auth:
                    auth_cand = l_auth
                if empty_cand is None and l_empty:
                    empty_cand = l_empty
        return None

    hit = _probe_list(COMMON_ENDPOINTS, timeout=1.2)
    if hit:
        return hit

    dynamic_eps = _discover_listening_endpoints(exclude_urls={e.rstrip("/") for e in COMMON_ENDPOINTS})
    if dynamic_eps:
        hit = _probe_list(dynamic_eps, timeout=1.0)
        if hit:
            return hit

    if auth_cand:
        raise AuthRequiredError(auth_cand[0], auth_cand[1])
    if empty_cand:
        raise RuntimeError(
            f"已发现本机服务 {empty_cand}，但当前未加载任何模型（/models 返回空列表），请先在客户端中加载并启动模型。"
        )
    raise RuntimeError(
        "未在本机常见端口及本地推理进程中发现已启动的 OpenAI 兼容服务。\n"
        "  · 如果你使用的是 Strata / LM Studio 等桌面客户端，请确认已加载模型并开启本地 API 服务；\n"
        "  · 如果使用了自定义端口，可在下方直接输入端口号（如 8080 或 17880）。"
    )


def fetch_models(endpoint: str, api_key: Optional[str] = None) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = http_json(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=5.0)
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


def _model_path_candidates(value: str) -> List[str]:
    """File name first, then up to 3 parent folder names (e.g. 'Qwen3.8-Flash-Next-GGUF/iq3_s.gguf').
    Used in memory ONLY for model matching; the path is never stored or uploaded."""
    parts = [p for p in (value or "").replace("\\", "/").split("/") if p and not p.endswith(":")]
    return list(reversed(parts[-4:]))


def detect_server_model_id(endpoint: str, model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    """Best-effort leaderboard model_id detection: checks model_name first, then local /props or /models metadata in memory ONLY."""
    mid = map_model_id(model_name)
    if mid:
        return mid
    if endpoint and is_local_endpoint(endpoint):
        ep = endpoint.rstrip("/")
        root = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            props = http_json(f"{root}/props", headers=headers, timeout=1.5)
            if isinstance(props, dict):
                for key in ("model_path", "model_alias", "model"):
                    v = props.get(key)
                    if isinstance(v, str) and v:
                        for cand in _model_path_candidates(v):
                            mid = map_model_id(cand)
                            if mid:
                                return mid
        except Exception:
            pass
        try:
            data = http_json(f"{ep}/models", headers=headers, timeout=1.5)
            items = data.get("data") if isinstance(data, dict) else None
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for key in ("name", "root", "owned_by", "model"):
                        v = item.get(key)
                        if isinstance(v, str) and v:
                            for cand in _model_path_candidates(v):
                                mid = map_model_id(cand)
                                if mid:
                                    return mid
        except Exception:
            pass
    return None


def _strip_model_file_ext(name: str) -> str:
    return re.sub(r"\.(?:gguf|bin|safetensors|mlx)$", "", name.strip(), flags=re.IGNORECASE)


def resolve_model_raw(
    endpoint: str,
    model_name: str,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Resolve a human-readable model_raw without ever exposing local directory paths.

    If API already returned a recognizable model name, keep it as-is.
    When API only returned a short alias (e.g. 'iq3_s'), prefer the pure filename stem
    from local /props or /models metadata; otherwise fall back to '<Label> (<alias>)'.
    """
    raw = (model_name or "").strip()
    if not raw or map_model_id(raw):
        return raw

    if endpoint and is_local_endpoint(endpoint):
        ep = endpoint.rstrip("/")
        root = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            props = http_json(f"{root}/props", headers=headers, timeout=1.5)
            if isinstance(props, dict):
                for key in ("model_path", "model_alias", "model"):
                    v = props.get(key)
                    if isinstance(v, str) and v:
                        stem = _strip_model_file_ext(os.path.basename(v.replace("\\", "/")))
                        if stem and map_model_id(stem):
                            return stem
        except Exception:
            pass
        try:
            data = http_json(f"{ep}/models", headers=headers, timeout=1.5)
            items = data.get("data") if isinstance(data, dict) else None
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for key in ("name", "root", "model"):
                        v = item.get(key)
                        if isinstance(v, str) and v:
                            stem = _strip_model_file_ext(os.path.basename(v.replace("\\", "/")))
                            if stem and map_model_id(stem):
                                return stem
        except Exception:
            pass

    if model_id:
        label = dict(KNOWN_LEADERBOARD_MODELS).get(model_id, model_id)
        if label and label.lower() != raw.lower():
            return f"{label} ({raw})"
    return raw


def detect_server_quant(endpoint: str, model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    """Best-effort quant detection: checks model_name first, then enriches from local /props filename in memory ONLY (path is never stored or uploaded)."""
    q_from_name = infer_quant(model_name)
    if q_from_name and (q_from_name.startswith("GSQ-RCO") or q_from_name.startswith("UD-")):
        return q_from_name
    if endpoint and is_local_endpoint(endpoint):
        ep = endpoint.rstrip("/")
        root = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            props = http_json(f"{root}/props", headers=headers, timeout=1.5)
            if isinstance(props, dict):
                # Inspect basename only in memory to detect GSQ-RCO / ISTA-DASLab / UD- prefix; never persist path
                for key in ("model_path", "model_alias"):
                    v = props.get(key)
                    if isinstance(v, str) and v:
                        base = os.path.basename(v.replace("\\", "/"))
                        q_prop = infer_quant(base)
                        if q_prop and (not q_from_name or len(q_prop) > len(q_from_name)):
                            return q_prop
        except Exception:
            pass
    return q_from_name


def detect_server_ctx_limit(endpoint: str, api_key: Optional[str] = None) -> Optional[int]:
    """Best-effort detection of the running server's configured context limit (n_ctx / max_model_len)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    ep = endpoint.rstrip("/")
    root = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)

    # 1. llama.cpp / Strata /props endpoint (read numeric n_ctx ONLY)
    try:
        props = http_json(f"{root}/props", headers=headers, timeout=2.0)
        if isinstance(props, dict):
            dgs = props.get("default_generation_settings")
            if isinstance(dgs, dict) and isinstance(dgs.get("n_ctx"), int) and dgs["n_ctx"] > 0:
                return int(dgs["n_ctx"])
            if isinstance(props.get("n_ctx"), int) and props["n_ctx"] > 0:
                return int(props["n_ctx"])
    except Exception:
        pass

    # 2. Ollama：/api/ps 返回当前已加载模型实际分配的 context_length（OpenAI 接口超长会被静默截断）
    o_root = ollama_native_root(endpoint, api_key=api_key)
    if o_root:
        try:
            ps = http_json(f"{o_root}/api/ps", headers=headers, timeout=2.0)
            models = ps.get("models") if isinstance(ps, dict) else None
            if isinstance(models, list):
                lens = [int(m["context_length"]) for m in models
                        if isinstance(m, dict) and isinstance(m.get("context_length"), int) and m["context_length"] > 0]
                if lens:
                    return max(lens)
        except Exception:
            pass

    # 3. vLLM / SGLang / llama.cpp /v1/models metadata
    try:
        data = http_json(f"{ep}/models", headers=headers, timeout=2.0)
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list) and items and isinstance(items[0], dict):
            m0 = items[0]
            for key in ("max_model_len", "context_length", "n_ctx"):
                v = m0.get(key)
                if isinstance(v, int) and v > 0:
                    return int(v)
            meta = m0.get("meta")
            if isinstance(meta, dict):
                for key in ("n_ctx", "n_ctx_train"):
                    v = meta.get(key)
                    if isinstance(v, int) and v > 0:
                        return int(v)
    except Exception:
        pass
    return None


def effective_ctx_target(tier_ctx: int, gen_tokens: int, ctx_limit: Optional[int]) -> Optional[int]:
    """Return actual prompt token target for a context tier, fitting within ctx_limit when tier == ctx_limit."""
    if not ctx_limit or ctx_limit <= 0:
        return tier_ctx
    margin = gen_tokens + 128
    if tier_ctx + margin <= ctx_limit:
        return tier_ctx
    if tier_ctx <= ctx_limit and ctx_limit > margin + 256:
        # e.g. server n_ctx=262144 and tier=256k -> trim slightly to fit gen_tokens + chat template
        return ctx_limit - margin
    return None


def build_warmup_prompt(ctx_limit: Optional[int] = None) -> str:
    """Build a short (~256-token) warmup prompt so prefill kernels and KV buffers warm up before 1k."""
    target = 256
    if ctx_limit and ctx_limit > 0:
        target = min(target, max(32, ctx_limit - 128))
    return build_prompt(target, "warmup")


def build_prompt(target_tokens: int, nonce: str, chars_per_token: float = 5.6) -> str:
    # Unique nonce at the beginning prevents KV prefix-cache reuse between runs.
    header = (
        f"[Bench run {nonce}] Read the reference passage below, then write a clear, "
        f"detailed technical explanation of how memory bandwidth affects local LLM inference "
        f"without stopping early.\n\n"
    )
    cpt = max(3.0, min(8.0, float(chars_per_token or 5.6)))
    target_chars = max(0, int(round(target_tokens * cpt)) - len(header))
    if target_chars <= 0:
        return header
    reps = (target_chars // len(FILLER_PARAGRAPH)) + 1
    body = (FILLER_PARAGRAPH * reps)[:target_chars]
    return header + body


def ollama_native_root(endpoint: str, api_key: Optional[str] = None) -> Optional[str]:
    """Return the Ollama native API root (e.g. http://127.0.0.1:11434) when the endpoint is Ollama, else None.

    Ollama's OpenAI-compatible /v1 stream carries no server-side timings, but its native /api/chat
    reports prompt_eval_duration / eval_duration measured inside the engine, so we prefer that.
    """
    ep = (endpoint or "").rstrip("/")
    if ep in _OLLAMA_NATIVE_ROOTS:
        return _OLLAMA_NATIVE_ROOTS[ep]
    root: Optional[str] = None
    if ep and is_local_endpoint(ep):
        cand = re.sub(r"/v1$", "", ep, flags=re.IGNORECASE)
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            data = http_json(f"{cand}/api/version", headers=headers, timeout=1.5)
            if isinstance(data, dict) and isinstance(data.get("version"), str) and data["version"]:
                root = cand
        except Exception:
            root = None
    _OLLAMA_NATIVE_ROOTS[ep] = root
    return root


def _run_ollama_native_completion(
    root: str,
    model_name: str,
    prompt: str,
    gen_tokens: int,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Stream one request through Ollama's native /api/chat and convert its engine timings."""
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0, "num_predict": gen_tokens},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{root}/api/chat", data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    final: Dict[str, Any] = {}
    with _urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                raise RuntimeError(f"Ollama error: {obj.get('error')}")
            msg = obj.get("message") or {}
            piece = (msg.get("content") or "") + (msg.get("thinking") or "") if isinstance(msg, dict) else ""
            if piece:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_end = now
            if obj.get("done"):
                final = obj
                break
    if t_end is None:
        t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("Server returned no generated tokens.")

    ttft_s = max(0.001, t_first - t_start)
    decode_s = max(0.001, t_end - t_first)
    eval_n = int(final.get("eval_count") or 0)
    eval_ns = float(final.get("eval_duration") or 0)
    prompt_n = int(final.get("prompt_eval_count") or 0)
    prompt_ns = float(final.get("prompt_eval_duration") or 0)
    completion_tokens = max(1, eval_n)
    prompt_tokens = prompt_n or max(1, round(len(prompt) / 4))
    sse_tg_tps = max(0.1, (max(1, completion_tokens - 1)) / decode_s) if completion_tokens > 1 else max(0.1, 1.0 / ttft_s)

    if eval_n > 0 and eval_ns > 0:
        tg_tps = eval_n / (eval_ns / 1e9)
        pp_tps = prompt_n / (prompt_ns / 1e9) if (prompt_n > 0 and prompt_ns > 0) else max(0.1, prompt_tokens / ttft_s)
        source = "timings"
    else:
        tg_tps = sse_tg_tps
        pp_tps = max(0.1, prompt_tokens / ttft_s)
        source = "sse"
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": completion_tokens,
        "tg_tps": round(tg_tps, 2),
        "pp_tps": round(pp_tps, 2),
        "sse_tg_tps": round(sse_tg_tps, 2),
        "ttft_ms": round(ttft_s * 1000.0, 1),
        "source": source,
        "tokens_estimated": eval_n <= 0,
    }


def run_single_completion(
    endpoint: str,
    model_name: str,
    prompt: str,
    gen_tokens: int,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    ttft_overhead_s: float = 0.0,
) -> Dict[str, Any]:
    ollama_root = ollama_native_root(endpoint, api_key=api_key)
    if ollama_root:
        try:
            return _run_ollama_native_completion(
                ollama_root, model_name, prompt, gen_tokens, api_key=api_key, timeout=timeout
            )
        except urllib.error.HTTPError as e:
            if e.code not in (404, 405, 501):
                raise
            # 原生接口不可用（反代只暴露了 /v1 等）：退回 OpenAI 兼容接口
            _OLLAMA_NATIVE_ROOTS[endpoint.rstrip("/")] = None

    url = f"{endpoint.rstrip('/')}/chat/completions"
    ep_key = endpoint.rstrip("/")
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # 让模型跑满 max_tokens，避免提前结束导致生成数过少（< 64 不计入天梯）。
    # llama.cpp / vLLM / SGLang 支持该参数；不支持且报 400/422 的服务会自动去掉重试。
    use_ignore_eos = ep_key not in _NO_IGNORE_EOS_ENDPOINTS
    if use_ignore_eos:
        body["ignore_eos"] = True
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    chunk_tokens = 0
    out_chars = 0
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}

    try:
        resp_cm = _urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if use_ignore_eos and e.code in (400, 422):
            _NO_IGNORE_EOS_ENDPOINTS.add(ep_key)
            return run_single_completion(
                endpoint, model_name, prompt, gen_tokens, api_key=api_key, timeout=timeout, ttft_overhead_s=ttft_overhead_s
            )
        raise

    with resp_cm as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
            if isinstance(obj.get("usage"), dict) and obj["usage"]:
                usage = obj["usage"]

            choices = obj.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_end = now
                    chunk_tokens += 1
                    out_chars += len(piece)

    if t_end is None:
        t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("Server returned no generated tokens.")

    ttft_s = max(0.001, t_first - t_start)
    decode_s = max(0.001, t_end - t_first)

    prompt_tokens = int(
        timings.get("prompt_n")
        or usage.get("prompt_tokens")
        or max(1, round(len(prompt) / 4))
    )
    server_completion = timings.get("predicted_n") or usage.get("completion_tokens")
    tokens_estimated = not server_completion
    if server_completion:
        completion_tokens = int(server_completion)
    else:
        # 服务端未返回 token 数：只按数据包计数会在「一包多 token」（MTP/推测解码/批量推送）时严重低估速度，
        # 因此同时按输出字符数粗估（英文约 4 字符/token），取两者较大值。
        completion_tokens = max(1, chunk_tokens, int(round(out_chars / 4.0)))

    # SSE decode speed: tokens after the first token divided by decode interval.
    sse_tg_tps = max(0.1, (max(1, completion_tokens - 1)) / decode_s) if completion_tokens > 1 else max(0.1, 1.0 / ttft_s)
    # 无服务端计时时只能用首字时间估算读题速度。首字时间里混有网络、排队、套模板与首 token 生成等固定开销，
    # 短上下文时占比很大（1k 档可低估 40% 以上）。若已测得固定开销，先扣除再估算；
    # 扣除后剩余时间过短（开销测量不稳或读题极快）时退回原算法，避免算出离谱的大数。
    prefill_s = ttft_s
    if ttft_overhead_s > 0:
        adjusted = ttft_s - ttft_overhead_s
        if adjusted >= 0.2 * ttft_s:
            prefill_s = adjusted
    sse_pp_tps = max(0.1, prompt_tokens / prefill_s)

    if timings.get("predicted_per_second"):
        tg_tps = float(timings["predicted_per_second"])
        pp_tps = float(timings["prompt_per_second"]) if timings.get("prompt_per_second") else sse_pp_tps
        source = "timings"
    else:
        tg_tps = sse_tg_tps
        pp_tps = sse_pp_tps
        source = "sse"

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": completion_tokens,
        "tg_tps": round(tg_tps, 2),
        "pp_tps": round(pp_tps, 2),
        "sse_tg_tps": round(sse_tg_tps, 2),
        "ttft_ms": round(ttft_s * 1000.0, 1),
        "source": source,
        "tokens_estimated": tokens_estimated and source != "timings",
        **_draft_stats(timings),
    }


def _draft_stats(timings: Dict[str, Any]) -> Dict[str, Any]:
    """llama.cpp-style servers report draft_n / draft_n_accepted when MTP or speculative decoding is on."""
    try:
        n = int(timings.get("draft_n") or 0)
        acc = int(timings.get("draft_n_accepted") or 0)
    except (TypeError, ValueError):
        return {}
    if n <= 0:
        return {}
    return {"draft_n": n, "draft_accepted": max(0, min(acc, n))}


# 默认开启 MTP、且 MTP 开关对成绩影响极大的引擎：必须记录 MTP 状态，天梯图才能分开排
MTP_SENSITIVE_ENGINES = ("strata", "basalt")


def infer_mtp_from_results(results: List[Dict[str, Any]]) -> Optional[bool]:
    """True when the server reported draft tokens (MTP / speculative decoding active); otherwise unknown."""
    return True if any((r or {}).get("draft_accept_rate") is not None for r in results) else None


def measure_ttft_overhead(
    endpoint: str, model_name: str, api_key: Optional[str] = None, probes: int = 3
) -> Optional[float]:
    """Median first-token latency of a near-empty prompt = fixed per-request overhead (seconds).

    Only meaningful for engines without server-side timings. Returns None when the server does
    report timings (no correction needed) or the probe fails.
    """
    vals: List[float] = []
    for i in range(max(1, probes)):
        try:
            r = run_single_completion(endpoint, model_name, f"Hi {i}", 4, api_key=api_key, timeout=60.0)
        except Exception:
            return None
        if r.get("source") == "timings":
            return None
        vals.append(float(r["ttft_ms"]) / 1000.0)
    return statistics.median(vals) if vals else None


def tier_quality_warnings(samples: List[Dict[str, Any]], gen_tokens_cap: int) -> List[str]:
    """Human-readable warnings for a finished context tier (shared by CLI and WebUI)."""
    warns: List[str] = []
    if not samples:
        return warns
    med_gen = int(round(statistics.median(s["gen_tokens"] for s in samples)))
    if med_gen < LADDER_MIN_GEN_TOKENS:
        warns.append(
            f"模型每轮只生成了约 {med_gen} 个 token 就提前结束（天梯要求 ≥{LADDER_MIN_GEN_TOKENS}），"
            f"本档成绩可以上传但不会计入速度天梯。可换用不会过早停止的模型/对话模板，或确认服务端支持 ignore_eos。"
        )
    elif med_gen < min(gen_tokens_cap, 256) * 0.5:
        warns.append(f"模型每轮只生成了约 {med_gen} 个 token（上限 {gen_tokens_cap}），样本偏短，成绩波动可能较大。")
    if any(s.get("tokens_estimated") for s in samples):
        warns.append(
            "推理服务未返回生成 token 数（usage），已按输出字数估算生成速度，结果可能有偏差；"
            "建议升级推理服务或使用支持 stream_options.include_usage 的版本。"
        )
    return warns


LogFn = Callable[..., None]


def _stderr_log(msg: str, color: Optional[str] = None) -> None:
    print(_c(msg, color) if color else msg, file=sys.stderr)


def benchmark_context(
    endpoint: str,
    model_name: str,
    ctx_prompt: int,
    gen_tokens: int,
    runs: int,
    api_key: Optional[str] = None,
    chars_per_token: float = 5.6,
    tier_label: Optional[str] = None,
    log: Optional[LogFn] = None,
    timeout: Optional[float] = None,
    ttft_overhead_s: float = 0.0,
) -> Tuple[Dict[str, Any], float]:
    """Run one context tier (median of N runs). Shared by the CLI and the local WebUI."""
    emit = log or _stderr_log
    samples: List[Dict[str, Any]] = []
    tier_runs = effective_runs_for_ctx(ctx_prompt, runs)
    req_timeout = timeout or timeout_for_ctx(ctx_prompt)
    ctx_lbl = tier_label or format_ctx_label(ctx_prompt)
    cpt = chars_per_token
    emit(f"\n>> 开始测速 [{ctx_lbl} 上下文]（输入 ≈{ctx_prompt} tok，每轮生成上限 {gen_tokens} tok，共 {tier_runs} 轮）：", "1;36")
    for i in range(tier_runs):
        nonce = f"{ctx_prompt}-{i}-{int(time.time() * 1000) % 100000}"
        prompt = build_prompt(ctx_prompt, nonce, chars_per_token=cpt)
        if ctx_prompt >= 16384:
            emit(f"   第 {i + 1}/{tier_runs} 轮进行中（长上下文读题较慢，请耐心等待）...")
        r = run_single_completion(
            endpoint, model_name, prompt, gen_tokens, api_key=api_key, timeout=req_timeout, ttft_overhead_s=ttft_overhead_s
        )
        samples.append(r)
        if r["prompt_tokens"] >= 200:
            measured_cpt = len(prompt) / float(r["prompt_tokens"])
            if 3.0 <= measured_cpt <= 8.0:
                cpt = measured_cpt
        src_lbl = "服务端精确计时" if r["source"] == "timings" else "客户端流式计时"
        emit(
            f"   第 {i + 1}/{tier_runs} 轮：生成 {_c(str(r['tg_tps']), '1;33') if log is None else r['tg_tps']} tok/s | "
            f"读题 {r['pp_tps']} tok/s | 首字 {r['ttft_ms']} ms（输入 {r['prompt_tokens']} / 生成 {r['gen_tokens']} tok，{src_lbl}）"
            + (f" | MTP 接受率 {r['draft_accepted'] / r['draft_n']:.0%}" if r.get("draft_n") else "")
        )

    for w in tier_quality_warnings(samples, gen_tokens):
        emit(f"   ! {w}", "33")

    res = {
        "ctx_prompt": int(round(statistics.median(s["prompt_tokens"] for s in samples))),
        "gen_tokens": int(round(statistics.median(s["gen_tokens"] for s in samples))),
        "tg_tps": round(statistics.median(s["tg_tps"] for s in samples), 2),
        "pp_tps": round(statistics.median(s["pp_tps"] for s in samples), 2),
        "sse_tg_tps": round(statistics.median(s["sse_tg_tps"] for s in samples), 2),
        "ttft_ms": round(statistics.median(s["ttft_ms"] for s in samples), 1),
        "runs": tier_runs,
        "source": "timings" if all(s["source"] == "timings" for s in samples) else "sse",
    }
    draft_n = sum(int(s.get("draft_n") or 0) for s in samples)
    if draft_n > 0:
        res["draft_accept_rate"] = round(sum(int(s.get("draft_accepted") or 0) for s in samples) / draft_n, 3)
    return res, cpt


# 单个请求的超时上限：再慢的机器也不让一轮等超过 2 小时
MAX_REQUEST_TIMEOUT = 7200.0


def estimate_tier_timeout(target_ctx: int, prev: Optional[Dict[str, Any]]) -> Tuple[float, Optional[float]]:
    """Return (socket timeout, estimated prefill seconds) for a tier, scaled by the previous tier's speed.

    Most servers send nothing until the whole prompt is processed, so the read timeout must cover
    the prefill. Prefill speed drops at long context (attention cost), hence the 0.5 safety factor.
    """
    base = timeout_for_ctx(target_ctx)
    if not prev or not prev.get("pp_tps") or not prev.get("ctx_prompt"):
        return base, None
    pp = float(prev["pp_tps"])
    if pp <= 0:
        return base, None
    est = target_ctx / (pp * 0.5)
    return min(MAX_REQUEST_TIMEOUT, max(base, est * 1.5 + 120.0)), est


def run_benchmark_tiers(
    endpoint: str,
    model_name: str,
    ctx_list: List[int],
    gen_tokens: int,
    runs: int,
    api_key: Optional[str] = None,
    ctx_limit: Optional[int] = None,
    warmup: bool = True,
    log: Optional[LogFn] = None,
) -> List[Dict[str, Any]]:
    """Warm up, then benchmark each context tier in order. Shared by the CLI and the local WebUI.

    A failing tier after at least one success keeps the finished tiers and stops (usually the
    server's context/VRAM limit); a failure on the very first tier is raised to the caller.
    """
    emit = log or _stderr_log
    if server_busy(endpoint, api_key=api_key):
        emit(
            "! 推理服务当前正在处理其他请求（例如有人正在聊天或有后台任务），测出的速度会偏低；"
            "建议先停止其它请求后再测。",
            "33",
        )
    warm_source: Optional[str] = None
    if warmup:
        emit("\n▸ 正在发送预热请求（避免冷启动影响成绩）...")
        warm_source = run_single_completion(
            endpoint, model_name, build_warmup_prompt(ctx_limit), 24, api_key=api_key
        ).get("source")

    overhead = 0.0
    if warm_source != "timings":
        measured = measure_ttft_overhead(endpoint, model_name, api_key=api_key)
        if measured:
            overhead = measured
            emit(
                f"▸ 该推理服务未提供服务端计时：已测得单次请求固定开销约 {overhead * 1000:.0f} ms，"
                f"估算读题速度时将扣除这部分开销。"
            )

    results: List[Dict[str, Any]] = []
    done: List[Tuple[str, int, float]] = []  # (档位标签, 实际输入目标, 超时)，与 results 一一对应，供异常重测使用
    cpt = 5.6
    for c in ctx_list:
        lbl = format_ctx_label(c)
        target = effective_ctx_target(c, gen_tokens, ctx_limit)
        if target is None:
            emit(f"▸ 跳过 [{lbl}]（当前服务上下文上限 {ctx_limit} tokens）")
            continue
        req_timeout, est = estimate_tier_timeout(target, results[-1] if results else None)
        if est is not None and est >= 120:
            emit(f"▸ [{lbl}] 预计每轮约 {est / 60:.0f} 分钟")
        try:
            res, cpt = benchmark_context(
                endpoint,
                model_name,
                target,
                gen_tokens,
                runs,
                api_key=api_key,
                chars_per_token=cpt,
                tier_label=lbl,
                log=log,
                timeout=req_timeout,
                ttft_overhead_s=overhead,
            )
            results.append(res)
            done.append((lbl, target, req_timeout))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if results:
                emit(f"! [{lbl}] 测速失败（可能超出当前上下文上限、显存不足或等待超时：{e}），已保留已完成档位成绩并停止更高档位。", "33")
                break
            raise

    # 异常重测：上下文越长生成越慢是常态；若某档比任何一个更长的档位还慢 3% 以上，多半是那一轮碰上了偶发波动
    # （后台任务、降频等）。全部测完后只重测这一档一次，取较好的一次。256k 等最长档只作参照、不重测。
    for i in range(len(results) - 1):
        cur = results[i]
        lbl, target, req_timeout = done[i]
        if target >= REMEASURE_MAX_CTX or not cur.get("tg_tps"):
            continue
        longer = [(done[j][0], results[j]["tg_tps"]) for j in range(i + 1, len(results)) if results[j].get("tg_tps")]
        if not longer:
            continue
        ref_lbl, ref_tg = max(longer, key=lambda x: x[1])
        if cur["tg_tps"] >= ref_tg * (1 - REMEASURE_TOLERANCE):
            continue
        emit(
            f"\n▸ [{lbl}] 生成速度（{cur['tg_tps']} tok/s）低于更长的 [{ref_lbl}]（{ref_tg} tok/s），"
            f"可能受偶发波动影响，正在重测该档一次..."
        )
        try:
            again, _ = benchmark_context(
                endpoint, model_name, target, gen_tokens, 1,
                api_key=api_key, chars_per_token=cpt, tier_label=lbl, log=log,
                timeout=req_timeout, ttft_overhead_s=overhead,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            emit(f"! [{lbl}] 重测未完成（{e}），保留原成绩。", "33")
            continue
        if again["tg_tps"] > cur["tg_tps"]:
            again["runs"] = int(cur.get("runs") or 1) + 1
            results[i] = again
            emit(f"   [{lbl}] 采用重测成绩 {again['tg_tps']} tok/s（原 {cur['tg_tps']} tok/s）")
        else:
            emit(f"   [{lbl}] 重测结果 {again['tg_tps']} tok/s，保留原成绩 {cur['tg_tps']} tok/s")
    return results


# 异常重测参数：比任一更长档位慢超过该比例才重测；输入 ≥192k（即 256k 档）不重测，避免多等一两分钟以上
REMEASURE_TOLERANCE = 0.03
REMEASURE_MAX_CTX = 196608


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="llm-speedtest：本机大模型标准化测速工具（https://itbanana.com/llm-speed）。"
    )
    p.add_argument("--endpoint", help="本机 OpenAI 兼容接口地址，例如 http://127.0.0.1:8080/v1 或端口号 8080（不填则自动扫描）")
    p.add_argument("--api-key", help="本机推理服务 API Key（选填，绝不上传）")
    p.add_argument("--model", help="指定要测试的模型名（默认取 /v1/models 返回的第一个模型）")
    p.add_argument("--model-id", help="标准模型 ID（例如 qwen38-flash-next、qwen38-27b）")
    p.add_argument("--quant", help="量化版本（例如 'GSQ-RCO IQ3_S'、'Q4_K_M'、'FP8'）")
    p.add_argument("--engine", help="推理引擎（strata / basalt / llamacpp / vllm / sglang / mlx / maya）")
    p.add_argument("--engine-version", help="推理引擎版本号（例如 0.2.15）")
    p.add_argument("--backend", help="计算后端（cuda / rocm / vulkan / metal）")
    p.add_argument("--kv-cache", help="KV Cache 精度（f16 / bf16 / fp8 / int8 / q8_0 / q4_0 / int4）")
    p.add_argument("--mtp", choices=["true", "false"], help="是否开启 MTP / 推测解码")
    p.add_argument("--ctx-limit", type=int, help="服务端配置的最大上下文长度（例如 131072；不填则自动探测）")
    p.add_argument("--fit", choices=["gpu", "hybrid"], help="模型加载方式：gpu（全显存）或 hybrid（显存+内存）")

    # Optional local hardware overrides (auto-detected by default)
    p.add_argument("--hw-kind", choices=["gpu", "unified"], help="覆盖硬件类型（gpu 或 unified）")
    p.add_argument("--hw-id", help="覆盖标准硬件 ID（例如 rtx4090-48）")
    p.add_argument("--hw-raw", help="覆盖硬件型号名称")
    p.add_argument("--gpu-count", type=int, help="覆盖显卡张数")
    p.add_argument("--vram-gb", type=int, help="覆盖单卡显存 (GB)")
    p.add_argument("--ram-gb", type=int, help="覆盖系统/统一内存 (GB)")
    p.add_argument("--ram-type", help="内存代数（DDR4 / DDR5 / DDR3 / Unified）")
    p.add_argument("--ram-channels", type=int, help="内存通道数（2 / 4 / 8 / 12）")
    p.add_argument("--os", dest="target_os", choices=["linux", "windows", "macos"], help="覆盖操作系统")

    # Benchmark parameters
    p.add_argument("--ctx", default=None, help=f"测试上下文档位，逗号分隔，支持 k 缩写（默认 {DEFAULT_CTX_STR}）")
    p.add_argument("--gen-tokens", type=int, default=256, help="每轮生成 token 上限（默认 256，各上下文档位统一）")
    p.add_argument("--runs", type=int, default=3, help="短上下文每档测速轮数，取中位数（默认 3；≥64k 自动 1 轮）")
    p.add_argument("--no-warmup", action="store_true", help="跳过预热请求")

    # Optional leaderboard identity & tuning note (voluntary public fields; anonymous if omitted)
    p.add_argument("--name", "--nickname", dest="nickname", help="选填：天梯图公开署名（≤20字，不含网址，留空匿名）")
    p.add_argument("--note", help="选填：一句话调优备注（≤120字，如 'Strata 0.2.15 + fa=on'）")
    p.add_argument("--anonymous", action="store_true", help="不附带本地随机匿名 client_id（完全匿名，不关联后续升级设备记录）")
    p.add_argument("--client-id", help="选填：手动指定本地匿名 client_id（换新机时可用于沿用旧记录）")

    # Mode / upload / interaction controls
    # WebUI 入口暂时隐藏（减少自由填写导致的脏数据），代码保留，仍可用 --ui 手动启动
    p.add_argument("--ui", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cli", action="store_true", help="直接进入终端命令行交互模式（默认即为此模式）")
    p.add_argument("--ui-port", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--detect-only", action="store_true", help="仅输出本机自动识别的硬件 JSON 并退出")
    p.add_argument("--dry-run", action="store_true", help="仅本地测速并打印明文 JSON，不上传")
    p.add_argument("--interactive", "-i", action="store_true", help="强制开启终端命令行交互模式")
    p.add_argument("--non-interactive", action="store_true", help="关闭所有交互提示（适合脚本与 AI Agent 调用）")
    p.add_argument("--yes", "-y", action="store_true", help="测速完成后跳过确认直接上传（命令行模式）")
    p.add_argument(
        "--resubmit",
        nargs="?",
        const="latest",
        metavar="JSON文件",
        help="重新上传之前保存的成绩（上传失败时使用；不填文件则取最近一次未上传成功的成绩）",
    )
    p.add_argument("--submit-url", default=DEFAULT_SUBMIT_URL, help=f"上传接口地址（默认 {DEFAULT_SUBMIT_URL}）")
    p.add_argument("--stats-url", default=DEFAULT_STATS_URL, help=f"社区统计接口地址（默认 {DEFAULT_STATS_URL}）")
    return p.parse_args(argv)


def resolve_hardware(args: argparse.Namespace) -> Dict[str, Any]:
    """Build local hardware spec from auto-detection with optional explicit overrides."""
    hw = detect_hardware()
    hw["detected_by"] = "manual" if (args.hw_id or args.hw_raw) else "local"

    if args.hw_kind:
        hw["kind"] = args.hw_kind
    if args.hw_id:
        hw["hw_id"] = args.hw_id.lower()
        if not args.hw_kind:
            hw["kind"] = "unified" if hw["hw_id"] in UNIFIED_IDS else "gpu"
        if not args.hw_raw and not hw.get("hw_raw"):
            hw["hw_raw"] = HW_ID_LABELS.get(hw["hw_id"], hw["hw_id"])
    if hw.get("kind") == "gpu" and hw.get("ram_type") == "Unified" and not args.ram_type:
        hw["ram_type"] = None
    if args.hw_raw:
        hw["hw_raw"] = args.hw_raw
    if args.gpu_count:
        hw["gpu_count"] = args.gpu_count
    if args.vram_gb:
        hw["vram_gb"] = args.vram_gb
        if not args.hw_kind and not hw.get("kind"):
            hw["kind"] = "gpu"
    if args.ram_gb:
        hw["ram_gb"] = args.ram_gb
    if args.ram_type:
        hw["ram_type"] = args.ram_type
    if args.ram_channels:
        hw["ram_channels"] = args.ram_channels
    if args.target_os:
        hw["os"] = args.target_os
    return hw


def _ask(prompt_text: str, default: str = "") -> str:
    suffix = f"（默认 {default}）" if default else ""
    try:
        ans = input(f"{prompt_text}{suffix}：").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return ans if ans else default


def _ask_int(prompt_text: str, default: Optional[int] = None) -> Optional[int]:
    ans = _ask(prompt_text, str(default) if default is not None else "")
    if not ans:
        return default
    try:
        return int(ans)
    except ValueError:
        return default


def _apply_parsed(args: argparse.Namespace, hw: Optional[Dict[str, Any]], parsed: Dict[str, Any]) -> None:
    if parsed.get("hw_id") and not args.hw_id:
        args.hw_id = parsed["hw_id"]
    if parsed.get("hw_raw") and not args.hw_raw:
        args.hw_raw = parsed["hw_raw"]
    if parsed.get("kind") and not args.hw_kind:
        args.hw_kind = parsed["kind"]
    if parsed.get("gpu_count") and not args.gpu_count:
        args.gpu_count = parsed["gpu_count"]
    if parsed.get("vram_gb") and not args.vram_gb:
        args.vram_gb = parsed["vram_gb"]
    if parsed.get("ram_gb") and not args.ram_gb:
        args.ram_gb = parsed["ram_gb"]
        if hw is not None and not hw.get("ram_gb"):
            hw["ram_gb"] = parsed["ram_gb"]
    if parsed.get("ram_type") and not args.ram_type:
        args.ram_type = parsed["ram_type"]
        if hw is not None and not hw.get("ram_type"):
            hw["ram_type"] = parsed["ram_type"]
    if parsed.get("ram_channels") and not args.ram_channels:
        args.ram_channels = parsed["ram_channels"]
        if hw is not None and not hw.get("ram_channels"):
            hw["ram_channels"] = parsed["ram_channels"]
    if parsed.get("os") and not args.target_os:
        args.target_os = parsed["os"]
        if hw is not None and not hw.get("os"):
            hw["os"] = parsed["os"]
    if parsed.get("engine") and not args.engine:
        args.engine = parsed["engine"]
    if parsed.get("quant"):
        args.quant = parsed["quant"]
    elif parsed.get("quant_prefix") and args.quant:
        base_q = re.sub(r"^(?:GSQ-RCO\s+|UD-)", "", args.quant, flags=re.IGNORECASE)
        if parsed["quant_prefix"] == "GSQ-RCO":
            args.quant = f"GSQ-RCO {base_q}"
        elif parsed["quant_prefix"] == "UD":
            args.quant = f"UD-{base_q}"
    if parsed.get("kv_cache") and not args.kv_cache:
        args.kv_cache = parsed["kv_cache"]
    if parsed.get("mtp") is not None and args.mtp is None:
        args.mtp = parsed["mtp"]


def print_welcome_banner() -> None:
    # Crisp pixel-block LLM-SPEEDTEST logo + ITBANANA.COM brand subtitle (fits <= 80 cols)
    logo_llm = [
        "█     █     █▄ ▄█ ",
        "█     █     █ █ █ ",
        "█     █     █   █ ",
        "▀▀▀▀▀ ▀▀▀▀▀ ▀   ▀ ",
    ]
    logo_dash = [
        "    ",
        "▄▄▄ ",
        "    ",
        "    ",
    ]
    logo_speedtest = [
        "▄▀▀▀▀ █▀▀▀▄ █▀▀▀▀ █▀▀▀▀ █▀▀▀▄ ▀▀█▀▀ █▀▀▀▀ ▄▀▀▀▀ ▀▀█▀▀",
        "▀▄▄▄  █▄▄▄▀ █▄▄▄  █▄▄▄  █   █   █   █▄▄▄  ▀▄▄▄    █  ",
        "    █ █     █     █     █   █   █   █         █   █  ",
        "▀▀▀▀  ▀     ▀▀▀▀▀ ▀▀▀▀▀ ▀▀▀▀    ▀   ▀▀▀▀▀ ▀▀▀▀    ▀  ",
    ]
    logo_rows = [
        f" {_c(l, '1;36')}{_c(d, '90')}{_c(s, '1;33')}"
        for l, d, s in zip(logo_llm, logo_dash, logo_speedtest)
    ]
    meta_row = (
        f" {_c('v' + CLIENT_VERSION, '1;36')}  {_c('·', '90')}  "
        f"{_c('by', '90')} {_c('ITBANANA.COM', '1;33')}  {_c('·', '90')}  "
        f"{_c('本地大模型速度天梯图官方实测工具', '1')}"
    )
    footer_rows = [
        f" {_c('●', '32')} 100% 开源 · 零依赖 · 仅采硬件、速度与随机匿名 client_id（--anonymous）",
        f" {_c('●', '32')} 绝不收集用户名/主机名/路径/原始 IP/API Key，上传前明文可见",
        f" {_c('▸', '36')} 速度天梯图：{_c('https://itbanana.com/llm-speed', '4;36')}",
    ]

    inner_width = max(77, max(_display_width(r) for r in logo_rows + [meta_row] + footer_rows) + 1)
    divider = " " + _c("─" * (inner_width - 2), "90")
    rows = logo_rows + ["", meta_row, divider] + footer_rows

    print("", file=sys.stderr)
    print(_c(f"╭{'─' * inner_width}╮", "36"), file=sys.stderr)
    print(_c("│", "36") + " " * inner_width + _c("│", "36"), file=sys.stderr)
    for row in rows:
        pad = inner_width - _display_width(row)
        print(_c("│", "36") + row + " " * pad + _c("│", "36"), file=sys.stderr)
    print(_c("│", "36") + " " * inner_width + _c("│", "36"), file=sys.stderr)
    print(_c(f"╰{'─' * inner_width}╯", "36"), file=sys.stderr)


def _read_nav_key(fd: Optional[int] = None) -> str:
    """Read a single navigation keypress from a TTY already in cbreak/raw mode (stdlib only)."""
    if platform.system().lower().startswith("win"):
        import msvcrt  # type: ignore

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            if code == "H":
                return "UP"
            if code == "P":
                return "DOWN"
            return ""
        if ch == "\x03":
            return "CTRL_C"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == "\x1b":
            return "ESC"
        if ch == "\t":
            return "TAB"
        return ch

    import select

    assert fd is not None
    raw = os.read(fd, 1)
    if not raw:
        return "CTRL_C"
    ch = raw.decode("utf-8", errors="ignore")
    if ch == "\x03":
        return "CTRL_C"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\t":
        return "TAB"
    if ch == "\x1b":
        if select.select([fd], [], [], 0.05)[0]:
            seq = os.read(fd, 1)
            if seq == b"[" and select.select([fd], [], [], 0.05)[0]:
                code = os.read(fd, 1)
                if code == b"A":
                    return "UP"
                if code == b"B":
                    return "DOWN"
        return "ESC"
    return ch


def prompt_launch_mode() -> str:
    """Let interactive terminal users choose CLI or local WebUI (with arrow-key support on TTY)."""
    options = [
        ("cli", "1", "终端直接测速", "推荐", "适合本机或 SSH 终端，1–2 次按键即刻开测"),
        ("ui", "2", "本地 WebUI 页面", "", "自动打开浏览器可视化面板，点选配置更直观"),
    ]

    def _render_menu(selected: int) -> List[str]:
        out: List[str] = []
        for idx, (_key, num, label, badge, desc) in enumerate(options):
            badge_str = f"  {_c('(' + badge + ')', '32')}" if badge else ""
            if idx == selected:
                out.append(f"  {_c('❯', '1;36')} {_c('[' + num + '] ' + label, '1;36')}{badge_str}")
                out.append(f"        {desc}")
            else:
                out.append(f"    {_c('[' + num + '] ' + label, '90')}{_c('  (' + badge + ')', '90') if badge else ''}")
                out.append(f"        {_c(desc, '90')}")
        return out

    # Interactive arrow-key selector when attached to a real terminal
    is_win = platform.system().lower().startswith("win")
    can_redraw = sys.stdin.isatty() and sys.stderr.isatty() and (is_win or os.environ.get("TERM", "xterm") != "dumb")
    if can_redraw:
        selected = 0
        print(
            f"\n  {_c('请选择启动方式', '1')}  {_c('（↑/↓ 切换 · Enter 确认 · 或直接按 1/2）', '90')}\n",
            file=sys.stderr,
        )
        lines = _render_menu(selected)
        rendered_height = len(lines)
        fd: Optional[int] = None
        old_settings = None
        try:
            if not is_win:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                # Disable ISIG so Ctrl+C arrives as \x03 cleanly
                raw_attrs = termios.tcgetattr(fd)
                raw_attrs[3] &= ~(termios.ISIG | termios.ECHO)
                termios.tcsetattr(fd, termios.TCSANOW, raw_attrs)

            sys.stderr.write("\033[?25l")
            sys.stderr.write("\n".join(lines) + "\n")
            sys.stderr.flush()

            while True:
                key = _read_nav_key(fd)
                if key in ("UP", "k", "K"):
                    selected = (selected - 1) % len(options)
                elif key in ("DOWN", "j", "J", "TAB"):
                    selected = (selected + 1) % len(options)
                elif key == "1":
                    selected = 0
                    break
                elif key == "2":
                    selected = 1
                    break
                elif key == "ENTER":
                    break
                elif key in ("CTRL_C", "ESC", "q", "Q"):
                    sys.stderr.write(f"\033[{rendered_height}A\033[J")
                    print(f"  {_c('已取消启动。', '90')}", file=sys.stderr)
                    return "quit"
                else:
                    continue

                sys.stderr.write(f"\033[{rendered_height}A\033[J")
                lines = _render_menu(selected)
                sys.stderr.write("\n".join(lines) + "\n")
                sys.stderr.flush()

            sys.stderr.write(f"\033[{rendered_height}A\033[J")
            lines = _render_menu(selected)
            sys.stderr.write("\n".join(lines) + "\n")
            chosen_key, _num, chosen_label, _b, _d = options[selected]
            print(f"\n  {_c('✓', '1;32')} 已选择：{_c(chosen_label, '1')}", file=sys.stderr)
            return chosen_key
        except Exception:
            # Fall back to standard text prompt if raw TTY fails
            pass
        finally:
            if not is_win and fd is not None and old_settings is not None:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stderr.write("\033[?25h")
            sys.stderr.flush()

    # Fallback for non-TTY / piped input / unit tests
    print(f"\n  {_c('请选择启动方式：', '1')}", file=sys.stderr)
    for line in _render_menu(0):
        print(line, file=sys.stderr)
    choice = _ask("\n  请输入 1 或 2，直接回车默认 1", "1").lower()
    if choice in ("2", "ui", "web", "webui", "w"):
        return "ui"
    if choice in ("q", "quit", "exit"):
        return "quit"
    return "cli"


def _prompt_api_key_for_endpoint(
    auth_ep: str, explicit_model: Optional[str] = None
) -> Optional[Tuple[str, List[str], str]]:
    """Interactively ask for API key when a local server returns HTTP 401/403."""
    print(f"\n! 已找到本机服务 {_c(auth_ep, '36')}，但你的推理服务开启了 API Key 验证。", file=sys.stderr)
    while True:
        entered = _ask("请输入本机服务的 API Key（仅本地验证，绝不上传；输入 q 退出）")
        if not entered or entered.lower() in ("q", "quit", "exit"):
            print("已退出测速。", file=sys.stderr)
            return None
        # Be forgiving if user pasted "--api-key sk-xxx"
        key = re.sub(r"^(?:--api-key|api-key=)\s*", "", entered, flags=re.IGNORECASE).strip()
        try:
            ep, models = resolve_endpoint(auth_ep, api_key=key, explicit_model=explicit_model)
            return ep, models, key
        except AuthRequiredError:
            print(_c("✗ API Key 不正确，请重新输入（输入 q 退出）。", "1;31"), file=sys.stderr)
        except RuntimeError as e:
            print(_c(f"✗ 连接失败：{e}", "1;31"), file=sys.stderr)
            return None


def run_interactive_wizard(
    args: argparse.Namespace,
    hw: Dict[str, Any],
    models: List[str],
    endpoint: str,
    detected_ctx_limit: Optional[int] = None,
) -> Tuple[str, str]:
    """Friendly Chinese CLI wizard: 1-2 keypresses to start benchmarking."""
    if args.model:
        model_name = args.model
    elif len(models) > 1:
        print("\n◆ 检测到本机已加载多个模型，请选择要测速的模型：", file=sys.stderr)
        for idx, m in enumerate(models, 1):
            print(f"   {idx}) {m}", file=sys.stderr)
        pick = _ask_int("请输入模型编号", 1) or 1
        model_name = models[min(max(1, pick), len(models)) - 1]
    else:
        model_name = models[0]

    if not args.engine:
        args.engine = infer_engine_auto(endpoint, model_name, api_key=args.api_key)
    if not args.quant:
        args.quant = detect_server_quant(endpoint, model_name, api_key=args.api_key)
    if not args.model_id:
        args.model_id = detect_server_model_id(endpoint, model_name, api_key=args.api_key)
    if not args.model_id:
        print(
            _c(
                f"\n! 接口返回的模型 ID 为「{model_name}」，无法自动匹配天梯图榜单模型（未匹配将无法显示在对应模型榜单中）。",
                "33",
            ),
            file=sys.stderr,
        )
        default_label = KNOWN_LEADERBOARD_MODELS[0][1]
        print(f"  请选择实际测试的模型编号（或直接输入模型名，直接回车默认 1) {default_label}）：", file=sys.stderr)
        for idx, (_mid, label) in enumerate(KNOWN_LEADERBOARD_MODELS, 1):
            print(f"   {idx:>2}) {label}", file=sys.stderr)
        pick_raw = _ask(f"模型编号或名称（直接回车 = 1 {default_label}）") or "1"
        if pick_raw:
            if pick_raw.isdigit():
                idx = int(pick_raw)
                if 1 <= idx <= len(KNOWN_LEADERBOARD_MODELS):
                    args.model_id = KNOWN_LEADERBOARD_MODELS[idx - 1][0]
            else:
                args.model_id = map_model_id(pick_raw) or pick_raw.strip().lower()
    guessed_q = args.quant
    ctx_str = args.ctx or DEFAULT_CTX_STR

    vram_part = f" {hw['vram_gb']}GB" if hw.get("vram_gb") and str(hw["vram_gb"]) not in str(hw.get("hw_raw", "")) else ""
    ram_part = f" + 内存 {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    ddr_part = f" {hw['ram_type']}" if hw.get("ram_type") else ""
    eng_part = f" | 引擎: {args.engine}" if args.engine else ""
    q_part = f" | 量化: {guessed_q}" if guessed_q else ""
    kv_part = f" | KV: {normalize_kv_cache(args.kv_cache)}" if args.kv_cache else ""
    lim_part = f"（服务上限 ≈{format_ctx_label(detected_ctx_limit)}，超限档位自动跳过）" if detected_ctx_limit else ""
    hw_label = f"{format_gpu_mix(hw['gpu_mix'])}（混搭）" if hw.get("gpu_mix") else f"{hw['hw_raw'] + vram_part} ×{hw['gpu_count']}"
    print(
        f"\n◆ 即将测速：{_c(hw_label, '1')}{ram_part}{ddr_part}"
        f" | 模型: {_c(model_name, '1')}{q_part}{eng_part}{kv_part}",
        file=sys.stderr,
    )
    print(f"▸ 测速上下文档位：{ctx_str}{lim_part}", file=sys.stderr)

    missing = []
    if not args.engine:
        missing.append("引擎如 Strata/llama.cpp")
    if not args.kv_cache:
        missing.append("KV如 kv int8/q8_0/fp8/f16")
    if hw.get("kind") == "gpu" and not hw.get("ram_type"):
        missing.append("内存如 DDR4/DDR5 双通道")
    if args.mtp is None and (args.engine or "").lower() in MTP_SENSITIVE_ENGINES:
        ans = _ask(f"{args.engine} 默认开启 MTP，开关对生成速度影响很大。本次是否开启了 MTP？[y=开启 / n=关闭 / 回车=不确定]").lower()
        if ans in ("y", "yes", "1", "开", "开启", "是"):
            args.mtp = "true"
        elif ans in ("n", "no", "0", "关", "关闭", "否"):
            args.mtp = "false"
    if args.mtp is None:
        missing.append("开MTP/不开MTP")
    if missing:
        extra = _ask(f"可选一句话补充（如 {'、'.join(missing)}；直接回车立即开始测速）")
        if extra:
            _apply_parsed(args, hw, parse_human_note(extra))

    return model_name, ctx_str


def print_summary(payload: Dict[str, Any]) -> None:
    hw = payload["hardware"]
    sw = payload["software"]
    mdl = payload["model"]
    ram_str = f"{hw.get('ram_gb') or '?'}GB" + (f" {hw['ram_type']}" if hw.get("ram_type") else "")
    vram_str = f" ({hw['vram_gb']}GB 显存)" if hw.get("vram_gb") else ""
    cpu_str = (
        f" · CPU {hw['cpu_raw']}"
        if hw.get("cpu_raw") and hw.get("kind") != "unified" and hw["cpu_raw"] != hw.get("hw_raw")
        else ""
    )
    kv_str = f" | KV Cache：{sw['kv_cache']}" if sw.get("kv_cache") else ""
    lines = [
        f"  ◆ 本机硬件：{hw['hw_raw']}{vram_str} ×{hw['gpu_count']}{cpu_str} · 内存 {ram_str} · {hw.get('os') or ''}",
        f"  ◆ 测试模型：{mdl['model_raw']} | 量化：{mdl.get('quant') or '未填'} | 引擎：{sw.get('engine') or '未填'}{kv_str}",
    ]
    if payload.get("nickname") or payload.get("note"):
        meta_parts = []
        if payload.get("nickname"):
            meta_parts.append(f"署名：{payload['nickname']}")
        if payload.get("note"):
            meta_parts.append(f"备注：{payload['note']}")
        lines.append(f"  ◆ {' | '.join(meta_parts)}")
    for r in payload["results"]:
        src_lbl = "服务端精确计时" if r["source"] == "timings" else "客户端流式计时"
        lbl = format_ctx_label(r["ctx_prompt"])
        lines.append(
            f"  ▸ [{lbl:>4}] ctx≈{r['ctx_prompt']} → 生成 {_c(str(r['tg_tps']) + ' tok/s', '1;33')} | "
            f"读题 {r.get('pp_tps') or '-'} tok/s | 首字 {r.get('ttft_ms') or '-'} ms（{r['runs']}轮，{src_lbl}）"
        )
    header_label = " 测速成绩单 "
    label_w = _display_width(header_label)
    inner_width = max(58, max((_display_width(line) for line in lines), default=0) + 1, label_w + 4)
    dash_total = inner_width - label_w
    dash_left = dash_total // 2
    dash_right = dash_total - dash_left
    print(_c(f"\n╭{'─' * dash_left}{header_label}{'─' * dash_right}╮", "36"), file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print(_c(f"╰{'─' * inner_width}╯", "36"), file=sys.stderr)


REVIEW_REASON_TEXT = {
    "outlier_vs_median": "明显高于同配置其他用户的成绩",
    "too_many_client_ids": "同一网络短时间内出现了多个匿名 ID",
    "outlier_vs_best": "明显高于同配置目前的最高成绩",
    "unknown_hw_id": "你的硬件型号还没有收录进硬件库",
    "over_bandwidth_cap": "超过了该硬件的理论显存带宽上限",
    "sse_timings_mismatch": "服务端计时与客户端计时差距过大",
    "manual_hardware": "硬件信息是手动指定的",
}


def review_notice(resp: Dict[str, Any]) -> List[str]:
    """Explain flagged (pending manual review) tiers and tiers the server dropped as invalid."""
    lines: List[str] = []
    res = resp.get("results") if isinstance(resp.get("results"), list) else []
    flagged = [r for r in res if isinstance(r, dict) and r.get("status") == "flagged"]
    if flagged:
        reasons: List[str] = []
        for r in flagged:
            key = str(r.get("flag_reason") or "").split(":", 1)[0]
            txt = REVIEW_REASON_TEXT.get(key, "触发了自动质检规则")
            if txt not in reasons:
                reasons.append(txt)
        tiers = "、".join(str(r.get("ctx_tier") or format_ctx_label(int(r.get("ctx_prompt") or 0))) for r in flagged)
        lines.append(
            f"成绩已收到，但 [{tiers}] 档因{'；'.join(reasons)}，需要人工审核通过后才会出现在速度天梯上（通常 1–2 天）。"
        )
    for n in resp.get("notices") if isinstance(resp.get("notices"), list) else []:
        if isinstance(n, str) and n.strip():
            lines.append(n.strip())
    dropped = resp.get("dropped") if isinstance(resp.get("dropped"), list) else []
    if dropped:
        lines.append(f"有 {len(dropped)} 档数据未通过服务端格式校验，未被收录：" + "；".join(str(d.get("error")) for d in dropped if isinstance(d, dict)))
    return lines


def get_results_dir() -> str:
    """Local folder for saved benchmark payloads (allows re-uploading after a network failure)."""
    return os.path.join(os.path.dirname(get_client_id_path()), "results")


def save_pending_payload(payload: Dict[str, Any]) -> Optional[str]:
    """Save the payload before uploading so a failed upload can be retried with --resubmit."""
    try:
        d = get_results_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("%Y%m%d-%H%M%S") + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except OSError:
        return None


def mark_uploaded(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.replace(path, path[: -len(".json")] + ".uploaded.json")
    except OSError:
        pass


def find_latest_pending() -> Optional[str]:
    d = get_results_dir()
    try:
        files = sorted(
            f for f in os.listdir(d) if f.endswith(".json") and not f.endswith(".uploaded.json")
        )
    except OSError:
        return None
    return os.path.join(d, files[-1]) if files else None


def server_busy(endpoint: str, api_key: Optional[str] = None) -> Optional[bool]:
    """llama.cpp-style /slots: True if any slot is processing another request, None if unknown."""
    root = re.sub(r"/v1/?$", "", endpoint.rstrip("/"))
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        data = http_json(f"{root}/slots", headers=headers, timeout=3.0)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    busy = False
    for slot in data:
        if not isinstance(slot, dict):
            continue
        if slot.get("is_processing") is True or slot.get("state") == 1:
            busy = True
    return busy


def community_text(resp: Dict[str, Any]) -> str:
    """'全站已汇聚 N 次真实测速（M 位贡献者），你是第 K 位加入的贡献者，感谢！' (falls back for older backends)."""
    total_c = resp.get("total_contributors")
    runs = resp.get("total_runs")
    no = resp.get("contributor_no")
    if runs and no:
        return f"全站已汇聚 {runs} 次真实测速（{total_c} 位贡献者），你是第 {no} 位加入的贡献者，感谢！"
    if runs:
        return f"全站已汇聚 {runs} 次真实测速，来自 {total_c} 位贡献者，感谢参与！"
    return f"全站已汇聚 {resp.get('total_samples') or 1} 条测速记录，来自 {total_c} 位贡献者，感谢参与！"


def personal_best_info(ranking: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise "is this upload a personal best?" across backend versions.

    New backend (>= 2026-11-27) ranks by each contributor's personal best and sends
    is_personal_best / vs_personal_best / personal_best(_tg) / personal_best_record_id.
    Older backends only sent gap_to_first, which at rank 1 meant "this run vs my best".
    """
    out: Dict[str, Any] = {
        "new_api": False, "is_pb": True, "below": 0.0, "best": None, "best_id": None,
        "has_prev_api": False, "prev": None, "gain": None, "was_top": False,
    }
    if not ranking:
        return out

    def _num(v: Any) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    if "is_personal_best" in ranking:
        out["new_api"] = True
        out["is_pb"] = bool(ranking.get("is_personal_best"))
        out["best"] = _num(ranking.get("personal_best_tg", ranking.get("personal_best")))
        vs_pb = _num(ranking.get("vs_personal_best"))
        if vs_pb is None and out["best"] is not None and _num(ranking.get("submitted_tg")) is not None:
            vs_pb = _num(ranking.get("submitted_tg")) - out["best"]
        out["below"] = round(max(0.0, -(vs_pb or 0.0)), 1) if not out["is_pb"] else 0.0
        out["best_id"] = ranking.get("personal_best_record_id")
        if "prev_personal_best" in ranking:
            # 提交前的个人最佳 / 是否原本就是同卡第一：区分首位开荒、刷新个人纪录、卫冕、新王登基
            out["has_prev_api"] = True
            out["prev"] = _num(ranking.get("prev_personal_best"))
            out["gain"] = _num(ranking.get("vs_prev_best"))
            out["was_top"] = bool(ranking.get("was_top"))
    else:
        gap = _num(ranking.get("gap_to_first")) or 0.0
        if ranking.get("same_hw_rank") == 1 and gap > 0.05:
            out["is_pb"] = False
            out["below"] = gap
    return out


def print_upload_ceremony(
    payload: Dict[str, Any],
    resp: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a celebratory trophy card in the terminal after a successful upload."""
    hw = payload.get("hardware") or {}
    sw = payload.get("software") or {}
    mdl = payload.get("model") or {}
    results = payload.get("results") or [{}]
    r0 = results[0]
    ranking = resp.get("ranking") if isinstance(resp.get("ranking"), dict) else None
    first_res = (resp.get("results") or [{}])[0] if isinstance(resp.get("results"), list) else {}
    rec_id = resp.get("record_id") or (first_res.get("id") if isinstance(first_res, dict) else None)

    same_hw_rank = ranking.get("same_hw_rank") if ranking else None
    same_hw_total = ranking.get("same_hw_total") if ranking else None
    gap = float(ranking.get("gap_to_first") or 0) if ranking else 0.0
    is_sole_hw = bool(ranking and same_hw_total is not None and same_hw_total <= 1)
    pb = personal_best_info(ranking)
    returning = pb["has_prev_api"] and pb["prev"] is not None  # 此前在该配置上已有成绩
    is_king = bool(ranking and same_hw_rank == 1 and not is_sole_hw and pb["is_pb"] and not (returning and pb["was_top"]))
    is_pioneer = bool(ranking and same_hw_rank == 1 and is_sole_hw and pb["is_pb"] and not returning)
    is_defend = bool(ranking and same_hw_rank == 1 and not is_sole_hw and pb["is_pb"] and returning and pb["was_top"])
    is_new_pb = bool(ranking and pb["is_pb"] and returning and (pb["gain"] or 0) > 0.05)
    gain_txt = f"+{pb['gain']:g} tok/s" if (pb["gain"] or 0) > 0.05 else ""

    review = review_notice(resp)
    has_flagged = any(
        isinstance(r, dict) and r.get("status") == "flagged" for r in (resp.get("results") or [])
    ) if isinstance(resp.get("results"), list) else False
    if has_flagged and not ranking:
        title = "★ 成绩已提交，等待人工审核后上榜 ★"
    elif is_king:
        title = "★ 新王登基！你的本机实测已登顶同配置同模型榜首！ ★"
    elif is_pioneer:
        title = "★ 首位开荒！你为该配置+模型立下了首条实测基准！ ★"
    elif is_defend:
        title = f"★ 卫冕榜首！你再次刷新了同配置同模型最高纪录（{gain_txt}） ★" if gain_txt else "★ 卫冕榜首！你仍是同配置同模型第一 ★"
    elif is_new_pb:
        title = f"★ 刷新个人纪录！{gain_txt}，成绩已载入速度天梯图 ★"
    elif ranking:
        title = "★ 恭喜上榜！实测成绩已载入速度天梯图 ★"
    else:
        title = "★ 恭喜！你的本机实测已成功上榜 ★"

    badge = f"实测编号 #{rec_id} · [本机实采认证]" if rec_id else "[本机实采认证]"
    cnt = f" ×{hw['gpu_count']}" if (hw.get("gpu_count") or 1) > 1 else ""
    cpu_part = (
        f" · {hw['cpu_raw']}"
        if hw.get("cpu_raw") and hw.get("kind") != "unified" and hw["cpu_raw"] != hw.get("hw_raw")
        else ""
    )
    ram_t = f" {hw['ram_type']}" if hw.get("ram_type") and hw.get("ram_type") != "Unified" else ""
    ram = f" · {hw['ram_gb']}GB{ram_t}" if hw.get("ram_gb") else (f" ·{ram_t}" if ram_t else "")
    model_raw = str(mdl.get("model_raw") or "-")
    quant_raw = str(mdl.get("quant") or "")
    quant = f" · {quant_raw}" if quant_raw and quant_raw.lower() != model_raw.strip().lower() else ""
    kv = f" · KV {sw['kv_cache']}" if sw.get("kv_cache") else ""
    author = payload.get("nickname") or "匿名贡献者"

    info_lines: List[str] = [
        f"  ★ {_c(badge, '1;36')}",
        f"  ◆ 硬件：{hw.get('hw_raw', 'Unknown')}{cnt}{cpu_part}{ram}",
        f"  ◆ 模型：{model_raw}{quant}{kv}",
    ]
    if len(results) == 1:
        info_lines.append(
            f"  ▸ 成绩：生成 {_c(str(r0.get('tg_tps', '-')) + ' tok/s', '1;32')}  |  "
            f"读题 {r0.get('pp_tps') or '-'} tok/s  |  署名：{author}"
        )
    else:
        info_lines.append(f"  ▸ 多档实测成绩（共 {len(results)} 档） · 署名：{author}")
        for r in results:
            lbl = format_ctx_label(r.get("ctx_prompt") or 0)
            info_lines.append(
                f"     [{lbl:>5}] 生成 {_c(str(r.get('tg_tps', '-')) + ' tok/s', '1;32')}  |  "
                f"读题 {r.get('pp_tps') or '-'} tok/s  |  首字 {r.get('ttft_ms') or '-'} ms"
            )
    if payload.get("note"):
        info_lines.append(f"  ◆ 备注：{payload['note']}")

    ranking_lines: List[str] = []
    if ranking:
        vs_med = ranking.get("vs_median_pct")
        med_val = ranking.get("same_hw_median")
        # 新版后端的排名/中位对比按「个人最佳」计；未刷新纪录时需明确说明，避免误读为本次成绩
        med_subject = "个人最佳较" if (pb["new_api"] and not pb["is_pb"]) else "较"
        med_txt = (
            f"（{med_subject}同配置中位 {med_val} tok/s {'+' if vs_med >= 0 else ''}{vs_med}%）"
            if (not is_sole_hw and vs_med is not None and med_val is not None)
            else ""
        )
        pb_tail = ""
        if not pb["is_pb"] and pb["below"] > 0.05:
            best_txt = f"（最佳 {pb['best']:g} tok/s）" if pb["best"] is not None else ""
            pb_tail = f" · 本次较个人历史最佳低 {pb['below']:g} tok/s{best_txt}，未刷新纪录"
        new_pb_tail = f" · ★ 刷新个人纪录 {gain_txt}" if (pb["is_pb"] and returning and gain_txt) else ""
        if same_hw_rank == 1:
            if not pb["is_pb"]:
                hw_tail = pb_tail
            elif is_sole_hw:
                hw_tail = (new_pb_tail + "（该配置目前仅你一人实测）") if returning else " · ★ 同配置首个基准！"
            elif returning and pb["was_top"]:
                hw_tail = " · ★ 卫冕同配置榜首" + (f"，刷新个人纪录 {gain_txt}" if gain_txt else "")
            else:
                hw_tail = " · ★ 当前同配置榜首！"
        else:
            # 旧后端：非第 1 名时 gap 即本次与第一名之差；新后端：个人最佳与第一名之差
            hw_tail = f" · 距同配置第一仅差 {gap:g} tok/s，调优后再冲一次！" if gap > 0.05 else ""
            hw_tail += pb_tail + new_pb_tail

        if same_hw_rank is not None and same_hw_total is not None:
            ranking_lines.append(
                f"  ★ 同配置同模型速度排名：第 {_c(str(same_hw_rank), '1;33')} / {same_hw_total} 名{med_txt}{hw_tail}"
            )
        same_model_rank = ranking.get("same_model_rank") or ranking.get("overall_rank")
        same_model_total = ranking.get("same_model_total") or ranking.get("overall_total")
        if same_model_rank is not None and same_model_total is not None:
            ranking_lines.append(
                f"  ★ 同模型速度排名：第 {_c(str(same_model_rank), '1;36')} / {same_model_total} 名"
            )

    community_lines: List[str] = [f"  ! {line}" for line in review]
    if resp.get("total_contributors"):
        community_lines.append(f"  ◆ {community_text(resp)}")
    if stats and isinstance(stats.get("group"), dict) and stats["group"].get("contributors"):
        g = stats["group"]
        if g.get("contributors") == 1 and g.get("tg_min") == g.get("tg_max"):
            community_lines.append(
                f"  ◆ 同配置同模型参考基准：{g['tg_median']} tok/s（共 1 位实测用户）"
            )
        else:
            community_lines.append(
                f"  ◆ 同配置同模型参考区间：{g['tg_min']}–{g['tg_max']} tok/s（中位 {g['tg_median']} tok/s，共 {g['contributors']} 位实测用户）"
            )

    ladder_url = (ranking or {}).get("ladder_url") or (
        f"https://itbanana.com/llm-speed?highlight={rec_id}" if rec_id else "https://itbanana.com/llm-speed"
    )
    footer_label = (
        f"直达天梯图查看本次成绩 #{rec_id}（天梯展示你的个人最佳 #{pb['best_id']}）"
        if (not pb["is_pb"] and pb["best_id"] and rec_id and pb["best_id"] != rec_id)
        else "直达天梯图高亮我的成绩"
    )

    title_w = _display_width(title)
    # 卡片宽度不超过终端宽度：过宽时终端会把每条边框折行，整张卡片就散了
    max_inner = max(title_w + 4, _terminal_columns() - 2)
    footer_line = f"  → {footer_label}：{_c(ladder_url, '1;4;36')}"
    footer_lines = [footer_line]
    if _display_width(footer_line) > max_inner:
        # 链接单独成行且不拆开，保证终端里可以直接点击
        footer_lines = _wrap_display(f"  → {footer_label}：", max_inner) + [f"     {_c(ladder_url, '1;4;36')}"]

    def _fit(lines: List[str]) -> List[str]:
        out: List[str] = []
        for ln in lines:
            out.extend(_wrap_display(ln, max_inner))
        return out

    info_lines = _fit(info_lines)
    ranking_lines = _fit(ranking_lines)
    community_lines = _fit(community_lines)

    all_body_lines = info_lines + ranking_lines + community_lines + footer_lines
    max_body_w = max((_display_width(line) for line in all_body_lines), default=0)
    inner_width = min(max(62, max_body_w + 1, title_w + 4), max(max_inner, title_w + 4))
    pad_total = inner_width - title_w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    print("", file=sys.stderr)
    print(_c(f"╔{'═' * inner_width}╗", "1;33"), file=sys.stderr)
    print(_c(f"║{' ' * pad_left}{title}{' ' * pad_right}║", "1;33"), file=sys.stderr)
    print(_c(f"╠{'═' * inner_width}╣", "33"), file=sys.stderr)
    for line in info_lines:
        print(line, file=sys.stderr)
    if ranking_lines:
        print(_c(f"╟{'─' * inner_width}╢", "33"), file=sys.stderr)
        for line in ranking_lines:
            print(line, file=sys.stderr)
    for line in community_lines:
        print(line, file=sys.stderr)
    print(_c(f"╟{'─' * inner_width}╢", "33"), file=sys.stderr)
    for line in footer_lines:
        print(line, file=sys.stderr)
    print(_c(f"╚{'═' * inner_width}╝", "1;33"), file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    configure_console()
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)

    if args.resubmit:
        return resubmit(args.resubmit, args.submit_url, args.stats_url)

    if args.detect_only:
        hw = resolve_hardware(args)
        print(json.dumps(hw, indent=2, ensure_ascii=False))
        return 0

    # Explicit --ui opens the local WebUI immediately
    if args.ui:
        import ui as ui_mod
        return ui_mod.serve_ui(
            sys.modules[__name__],
            submit_url=args.submit_url,
            stats_url=args.stats_url,
            port=args.ui_port,
        )

    interactive = args.interactive or (sys.stdin.isatty() and not args.yes and not args.non_interactive)

    print_welcome_banner()

    # 启动方式选单暂时隐藏：无参数运行直接进入终端测速（WebUI 只能用 --ui 手动打开）。
    # 恢复时把 SHOW_LAUNCH_MENU 改回 True 即可。
    if (
        SHOW_LAUNCH_MENU
        and len(raw_argv) == 0
        and sys.stdin.isatty()
        and not args.cli
        and not args.non_interactive
        and not args.yes
        and not args.dry_run
        and not args.interactive
    ):
        mode = prompt_launch_mode()
        if mode == "quit":
            return 0
        if mode == "ui":
            import ui as ui_mod
            return ui_mod.serve_ui(
                sys.modules[__name__],
                submit_url=args.submit_url,
                stats_url=args.stats_url,
                port=args.ui_port,
            )

    # 1. Detect local hardware first
    hw = resolve_hardware(args)
    vram_hint = f" ({hw['vram_gb']}GB 显存)" if hw.get("vram_gb") else ""
    cpu_hint = (
        f" · CPU {hw['cpu_raw']}"
        if hw.get("cpu_raw") and hw.get("kind") != "unified" and hw["cpu_raw"] != hw.get("hw_raw")
        else ""
    )
    ram_type_hint = f" {hw['ram_type']}" if hw.get("ram_type") and hw.get("ram_type") != "Unified" else ""
    ram_hint = f" · 内存 {hw['ram_gb']}GB{ram_type_hint}" if hw.get("ram_gb") else ""
    print(
        f"\n◆ 本机硬件：{_c(hw['hw_raw'] + vram_hint, '1')} ×{hw['gpu_count']}{cpu_hint}{ram_hint} · 系统 {hw.get('os') or ''}",
        file=sys.stderr,
    )
    if hw.get("detected_by") != "manual":
        for note in LAST_DETECT_NOTES:
            print(_c(f"  ! {note}", "33"), file=sys.stderr)

    # 与 WebUI 保持一致的参数范围
    args.runs = max(1, min(10, int(args.runs or 3)))
    args.gen_tokens = max(16, min(4096, int(args.gen_tokens or 256)))

    # 2. Discover or resolve local-only endpoint
    if args.endpoint:
        try:
            endpoint, models = resolve_endpoint(args.endpoint, api_key=args.api_key, explicit_model=args.model)
        except AuthRequiredError as e_auth:
            if not interactive:
                print(_c(f"\n✗ {e_auth}", "1;31"), file=sys.stderr)
                return 1
            res = _prompt_api_key_for_endpoint(e_auth.endpoint, explicit_model=args.model)
            if not res:
                return 1
            endpoint, models, args.api_key = res
        except ValueError as e:
            print(_c(f"\n✗ {e}", "1;31"), file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(_c(f"\n✗ {e}", "1;31"), file=sys.stderr)
            return 1
    else:
        print("▸ 正在扫描本机推理服务...", file=sys.stderr)
        try:
            endpoint, models = discover_endpoint(api_key=args.api_key)
        except AuthRequiredError as e_auth:
            if not interactive:
                print(_c(f"\n✗ {e_auth}", "1;31"), file=sys.stderr)
                return 1
            res = _prompt_api_key_for_endpoint(e_auth.endpoint, explicit_model=args.model)
            if not res:
                return 1
            endpoint, models, args.api_key = res
        except RuntimeError as e:
            if not interactive:
                print(_c(f"\n✗ {e}", "1;31"), file=sys.stderr)
                return 1
            print(f"\n! {e}", file=sys.stderr)
            while True:
                entered = _ask("请输入本机服务端口（如 8080，或完整本机地址；输入 q 退出）")
                if not entered or entered.lower() in ("q", "quit", "exit"):
                    print("已退出测速。", file=sys.stderr)
                    return 1
                try:
                    endpoint, models = resolve_endpoint(entered, api_key=args.api_key, explicit_model=args.model)
                    break
                except AuthRequiredError as e_auth:
                    res = _prompt_api_key_for_endpoint(e_auth.endpoint, explicit_model=args.model)
                    if not res:
                        return 1
                    endpoint, models, args.api_key = res
                    break
                except ValueError as e_local:
                    print(_c(f"✗ {e_local}", "1;31"), file=sys.stderr)
                except RuntimeError as e2:
                    print(_c(f"✗ {e2}", "1;31"), file=sys.stderr)

    if not is_local_endpoint(endpoint):
        print(_c(f"\n✗ {LOCAL_ONLY_ERR}", "1;31"), file=sys.stderr)
        return 2

    detected_ctx_limit = detect_server_ctx_limit(endpoint, api_key=args.api_key)
    effective_ctx_limit = args.ctx_limit or detected_ctx_limit
    ctx_lim_hint = f" · 上下文上限 {effective_ctx_limit}" if effective_ctx_limit else ""
    print(f"✓ 已连接本机服务：{_c(endpoint, '36')}（{len(models)} 个已加载模型{ctx_lim_hint}）", file=sys.stderr)

    if not models and not args.model:
        print(_c("❌ 未检测到已加载模型，请先加载模型或通过 --model <模型名> 指定。", "1;31"), file=sys.stderr)
        return 1

    if interactive:
        model_name, ctx_str = run_interactive_wizard(args, hw, models, endpoint, detected_ctx_limit=effective_ctx_limit)
    else:
        model_name = args.model or models[0]
        ctx_str = args.ctx or DEFAULT_CTX_STR
        print(f"▸ 测试模型：{_c(model_name, '1')}", file=sys.stderr)
        if not args.model and len(models) > 1:
            others = "、".join(models[1:6]) + ("…" if len(models) > 6 else "")
            print(
                _c(f"  ! 服务上加载了 {len(models)} 个模型，默认测第一个；如需测其它模型（{others}）请加 --model <模型名>", "33"),
                file=sys.stderr,
            )
        if not args.engine:
            args.engine = infer_engine_auto(endpoint, model_name, api_key=args.api_key)
        if not args.quant:
            args.quant = detect_server_quant(endpoint, model_name, api_key=args.api_key)
        if not args.model_id:
            args.model_id = detect_server_model_id(endpoint, model_name, api_key=args.api_key)

    results = run_benchmark_tiers(
        endpoint,
        model_name,
        parse_ctx_list(ctx_str),
        args.gen_tokens,
        args.runs,
        api_key=args.api_key,
        ctx_limit=effective_ctx_limit,
        warmup=not args.no_warmup,
    )

    if not results:
        print(_c("✗ 所有上下文档位均未能完成测速，请检查服务端上下文上限设置。", "1;31"), file=sys.stderr)
        return 1

    if args.mtp is None and infer_mtp_from_results(results):
        args.mtp = "true"
        print("▸ 推理服务返回了 MTP / 推测解码草稿统计，已自动记为「开启 MTP」。", file=sys.stderr)
    if args.mtp is None and (args.engine or "").lower() in MTP_SENSITIVE_ENGINES:
        print(
            _c(f"! {args.engine} 的 MTP 开关对成绩影响很大，但本次未能确认是否开启；"
               f"建议加 --mtp true/false 重新上传，否则天梯图无法与同配置成绩对比。", "33"),
            file=sys.stderr,
        )

    resolved_model_id = args.model_id or map_model_id(model_name)
    resolved_model_raw = resolve_model_raw(endpoint, model_name, model_id=resolved_model_id, api_key=args.api_key)
    client_id = None if args.anonymous else (clean_client_id(args.client_id) or get_or_create_client_id())
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "client_version": CLIENT_VERSION,
        **({"client_id": client_id} if client_id else {}),
        "hardware": hw,
        "software": {
            "engine": args.engine,
            "engine_version": args.engine_version,
            "backend": args.backend,
            "kv_cache": normalize_kv_cache(args.kv_cache),
            "mtp": True if args.mtp == "true" else (False if args.mtp == "false" else None),
            "ctx_limit": effective_ctx_limit,
        },
        "model": {
            "model_id": resolved_model_id,
            "model_raw": resolved_model_raw,
            "quant": args.quant or infer_quant(model_name),
            "fit": args.fit,
        },
        "results": results,
    }
    nickname = clean_public_text(args.nickname, 20)
    note = clean_public_text(args.note, 120)
    if nickname:
        payload["nickname"] = nickname
    if note:
        payload["note"] = note

    print_summary(payload)

    # In interactive mode (not dry-run / not --yes), allow optional nickname & tuning note before final JSON review
    if interactive and not args.dry_run and not args.yes and not nickname and not note:
        in_nick = clean_public_text(_ask("天梯图公开署名（选填，≤20字，直接回车保持匿名）"), 20)
        if in_nick:
            payload["nickname"] = in_nick
        in_note = clean_public_text(_ask("一句话调优备注（选填，≤120字，方便同卡抄作业，直接回车跳过）"), 120)
        if in_note:
            payload["note"] = in_note

    print("\n--- 即将上传的完整明文 JSON（零个人隐私，请核对） ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    saved_path = save_pending_payload(payload)

    if args.dry_run:
        print(_c("\n✓ 试运行完成（--dry-run 模式，未上传数据）。", "32"), file=sys.stderr)
        if saved_path:
            print(f"  成绩已保存到 {saved_path}，之后可用 --resubmit \"{saved_path}\" 上传。", file=sys.stderr)
        return 0

    if not args.yes:
        if not interactive:
            print("未传入 --yes，跳过上传（如需自动上传请加 --yes）。", file=sys.stderr)
            if saved_path:
                print(f"成绩已保存到 {saved_path}，之后可用 --resubmit 上传。", file=sys.stderr)
            return 0
        try:
            ans = input("\n确认将这份实测成绩上传至 itbanana.com 速度天梯图？[Y/n，直接回车立即上榜]：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消上传，本次测速成绩仅保留在本地。", file=sys.stderr)
            return 0
        if ans and ans not in ("y", "yes"):
            print("已取消上传，本次测速成绩仅保留在本地。", file=sys.stderr)
            if saved_path:
                print(f"  （已保存到 {saved_path}，改主意了可用 --resubmit 上传）", file=sys.stderr)
            return 0

    return upload_payload(payload, args.submit_url, args.stats_url, saved_path)


def upload_payload(
    payload: Dict[str, Any],
    submit_url: str,
    stats_url: Optional[str],
    saved_path: Optional[str] = None,
) -> int:
    """Upload a finished payload, then show the ranking card. Shared by normal runs and --resubmit."""
    print("\n▸ 正在将实测成绩刻入 itbanana.com 速度天梯图...", file=sys.stderr)
    retry_hint = (
        f"\n  成绩已保存在 {saved_path}，网络恢复后运行：python speedtest.py --resubmit" if saved_path else ""
    )
    try:
        resp = http_json(submit_url, payload=payload, timeout=30.0)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        try:
            detail = json.loads(detail).get("error") or detail
        except Exception:
            pass
        print(_c(f"\n✗ 上传失败（HTTP {e.code}）：{detail}", "1;31") + (retry_hint if e.code in (429, 500, 502, 503, 504) else ""), file=sys.stderr)
        return 1
    except Exception as e:
        print(_c(f"\n✗ 上传失败：{e}", "1;31") + retry_hint, file=sys.stderr)
        return 1

    mark_uploaded(saved_path)

    stats: Optional[Dict[str, Any]] = None
    hw = payload.get("hardware") or {}
    hw_id = hw.get("hw_id")
    model_id = (payload.get("model") or {}).get("model_id")
    if hw_id and model_id and stats_url:
        try:
            q = urllib.parse.urlencode({"hw_id": hw_id, "model_id": model_id, "gpu_count": hw.get("gpu_count") or 1})
            stats = http_json(f"{stats_url}?{q}", timeout=5.0)
        except Exception:
            pass

    print_upload_ceremony(payload, resp, stats=stats)
    return 0


def resubmit(target: str, submit_url: str, stats_url: Optional[str]) -> int:
    path = find_latest_pending() if target == "latest" else os.path.expanduser(target)
    if not path or not os.path.isfile(path):
        print(_c("✗ 没有找到待上传的成绩文件。", "1;31") + f"（保存目录：{get_results_dir()}）", file=sys.stderr)
        return 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        print(_c(f"✗ 读取成绩文件失败：{e}", "1;31"), file=sys.stderr)
        return 1
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        print(_c("✗ 这不是 llm-speedtest 保存的成绩文件。", "1;31"), file=sys.stderr)
        return 1
    print(f"▸ 重新上传 {path}", file=sys.stderr)
    return upload_payload(payload, submit_url, stats_url, saved_path=path)


def cli(argv: Optional[List[str]] = None) -> int:
    """Entry point: run main() and turn unexpected errors into short Chinese hints instead of tracebacks."""
    debug = os.environ.get("LLM_SPEEDTEST_DEBUG", "").strip() not in ("", "0", "false")
    try:
        return main(argv)
    except KeyboardInterrupt:
        print(_c("\n已中断测速（Ctrl+C），本次未上传任何数据。", "33"), file=sys.stderr)
        return 130
    except urllib.error.HTTPError as e:
        if debug:
            raise
        try:
            detail = e.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            detail = ""
        detail = re.sub(r"\s+", " ", detail)[:300]
        print(
            _c(f"\n✗ 推理服务返回错误 HTTP {e.code}{('：' + detail) if detail else ''}", "1;31")
            + "\n  请确认模型已正确加载、上下文上限足够，或查看推理服务自身的日志。",
            file=sys.stderr,
        )
        return 1
    except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout) as e:
        if debug:
            raise
        reason = getattr(e, "reason", None) or e
        print(
            _c(f"\n✗ 连接推理服务失败或请求超时：{reason}", "1;31")
            + "\n  请确认推理服务仍在运行；长上下文档位耗时较久，显存/内存不足时服务也可能中途退出。",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        if debug:
            raise
        print(
            _c(f"\n✗ 出现意外错误：{type(e).__name__}: {e}", "1;31")
            + "\n  如需反馈，请设置环境变量 LLM_SPEEDTEST_DEBUG=1 重新运行以获取完整报错信息。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
