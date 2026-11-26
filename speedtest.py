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
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

CLIENT_VERSION = "0.1.0"
SCHEMA_VERSION = 1
DEFAULT_SUBMIT_URL = "https://itbanana.com/api/llm-speedtest/submit"
DEFAULT_STATS_URL = "https://itbanana.com/api/llm-speedtest/stats"

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
    if re.search(r"\bstrata\b", low):
        return "strata"
    if re.search(r"\b(?:llama\.?cpp|llamacpp|llama-server|ollama|lm\s*studio)\b", low):
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


def infer_engine_auto(endpoint: str, model_name: str) -> Optional[str]:
    """Best-effort engine guess from port or model name."""
    try:
        port = urllib.parse.urlparse(endpoint).port
    except Exception:
        port = None
    if port in (11434, 1234):
        return "llamacpp"
    if "gsq-rco" in (model_name or "").lower():
        return "strata"
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

    if re.search(r"(?:不开|关闭|关|no\s*|without\s*)mtp|mtp\s*=\s*(?:false|0|off)", low):
        out["mtp"] = "false"
    elif re.search(r"(?:开|开启)?\bmtp\b|开\s*mtp|mtp\s*=\s*(?:true|1|on)", low):
        out["mtp"] = "true"

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


def parse_human_hardware(text: str) -> Dict[str, Any]:
    """Parse a single casual line like '4090 48G + DDR4 64G Strata', '双 7900 XTX 128G', 'M5 Ultra 256G'."""
    raw = (text or "").strip()
    if not raw:
        return {}
    low = raw.lower()
    note = parse_human_note(raw)

    if low in HW_ID_LABELS or low in UNIFIED_IDS:
        is_uni = low in UNIFIED_IDS
        out: Dict[str, Any] = {
            "kind": "unified" if is_uni else "gpu",
            "hw_id": low,
            "hw_raw": HW_ID_LABELS.get(low, low),
        }
        if not is_uni and low in HW_DEFAULT_VRAM:
            out["vram_gb"] = HW_DEFAULT_VRAM[low]
        if is_uni:
            out["ram_type"] = "Unified"
            if low.startswith("m"):
                out["os"] = "macos"
        return out

    # Detect GPU count from casual forms: "双 3090", "4卡 5090", "2x 7900 XTX", "7900xtx*2"
    gpu_count = None
    if re.search(r"(?:^|\s)双\s*", raw):
        gpu_count = 2
    else:
        m_cnt = re.search(r"(?:^|\s)(\d+)\s*(?:卡|张|[xX×*])\s*", raw) or re.search(r"[xX×*]\s*(\d+)(?:\s|$)", raw)
        if m_cnt:
            c = int(m_cnt.group(1))
            if 1 <= c <= 16:
                gpu_count = c

    # Remove explicit RAM segment before guessing GPU VRAM size so '5080 + 128G DDR5' doesn't treat 128 as VRAM
    gpu_part = re.split(r"[+＋]", low, maxsplit=1)[0]
    gpu_part_no_ram = re.sub(
        r"(?:(?:ddr[345]|lpddr5x?|内存|ram)\s*\d{2,4}\s*g(?:b)?|\d{2,4}\s*g(?:b)?\s*(?:ddr[345]|lpddr5x?|内存|ram))",
        " ",
        gpu_part,
    )
    all_gbs = [int(x) for x in re.findall(r"(\d{2,4})\s*g(?:b)?\b", gpu_part_no_ram)]

    # Normalize common shorthands before mapping
    norm = re.sub(r"4080\s*s\b", "4080 super", gpu_part)
    norm = re.sub(r"4070\s*ti\s*s\b", "4070 ti super", norm)
    norm = re.sub(r"4070\s*s\b", "4070 super", norm)
    norm = re.sub(r"7900\s*xtx", "7900 xtx", norm)
    norm = re.sub(r"7900\s*xt\b", "7900 xt", norm)
    norm = re.sub(r"2080\s*ti", "2080 ti", norm)
    norm = re.sub(r"3080\s*ti", "3080 ti", norm)
    norm = re.sub(r"5070\s*ti", "5070 ti", norm)
    norm = re.sub(r"5060\s*ti", "5060 ti", norm)
    norm = re.sub(r"4060\s*ti", "4060 ti", norm)

    # Decide VRAM candidate vs leftover RAM candidate
    vram_cand = None
    extra_ram = None
    if len(all_gbs) >= 2:
        vram_cand, extra_ram = all_gbs[0], all_gbs[1]
    elif len(all_gbs) == 1:
        g0 = all_gbs[0]
        # Modded/variant VRAM sizes are <= 96GB; e.g. '5080 128G' means 128G RAM
        if g0 <= 96 and not ("7900" in norm and g0 >= 64) and not ("3090" in norm and g0 >= 64) and not ("5080" in norm and g0 >= 64) and not ("5090" in norm and g0 >= 64) and not ("r9700" in norm and g0 >= 64):
            vram_cand = g0
        else:
            extra_ram = g0

    hw_id = None
    if "ai max" in norm and "395" in norm:
        hw_id = "aimax395"
    elif ("dgx" in norm and "spark" in norm) or "gb10" in norm:
        hw_id = "dgxspark"
    else:
        hw_id = map_apple_id(norm, None) or map_gpu_id(norm, vram_cand)

    is_uni = bool(hw_id and hw_id in UNIFIED_IDS)
    result: Dict[str, Any] = dict(note)
    result["kind"] = "unified" if is_uni else "gpu"
    result["hw_id"] = hw_id
    result["hw_raw"] = HW_ID_LABELS.get(hw_id, raw) if hw_id else raw
    if gpu_count:
        result["gpu_count"] = gpu_count
    if is_uni:
        result["ram_type"] = "Unified"
        if not result.get("ram_gb"):
            if all_gbs:
                result["ram_gb"] = all_gbs[0]
            elif extra_ram:
                result["ram_gb"] = extra_ram
        if hw_id and hw_id.startswith("m"):
            result["os"] = "macos"
    else:
        if vram_cand:
            result["vram_gb"] = vram_cand
        elif hw_id and hw_id in HW_DEFAULT_VRAM:
            result["vram_gb"] = HW_DEFAULT_VRAM[hw_id]
        if not result.get("ram_gb") and extra_ram:
            result["ram_gb"] = extra_ram
    return result


def detect_hardware_via_ssh(ssh_target: str) -> Optional[Dict[str, Any]]:
    """Read-only hardware spec detection over SSH (no hostname, username, or paths collected)."""
    remote_cmd = (
        "echo OS=$(uname -s); echo ARCH=$(uname -m); "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/^/NV=/'; "
        "fi; "
        "if [ -r /proc/meminfo ]; then grep '^MemTotal:' /proc/meminfo | awk '{print \"MEMKB=\"$2}'; fi; "
        "if [ \"$(uname -s)\" = \"Darwin\" ]; then "
        "  echo CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null); "
        "  echo MEMB=$(sysctl -n hw.memsize 2>/dev/null); "
        "  system_profiler SPDisplaysDataType 2>/dev/null | grep 'Total Number of Cores' | head -n 1 | sed 's/^/CORES=/'; "
        "fi"
    )
    out = run_cmd(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target, remote_cmd], timeout=10.0)
    if not out:
        return None

    os_raw = ""
    arch = ""
    nv_lines: List[str] = []
    mem_kb = 0
    mem_b = 0
    chip = ""
    gpu_cores = None

    for line in out.splitlines():
        if line.startswith("OS="):
            os_raw = line[3:].strip().lower()
        elif line.startswith("ARCH="):
            arch = line[5:].strip().lower()
        elif line.startswith("NV="):
            nv_lines.append(line[3:].strip())
        elif line.startswith("MEMKB=") and line[6:].strip().isdigit():
            mem_kb = int(line[6:].strip())
        elif line.startswith("MEMB=") and line[5:].strip().isdigit():
            mem_b = int(line[5:].strip())
        elif line.startswith("CHIP="):
            chip = line[5:].strip()
        elif line.startswith("CORES="):
            m = re.search(r"(\d+)", line)
            if m:
                gpu_cores = int(m.group(1))

    os_name = "macos" if os_raw == "darwin" else ("windows" if os_raw.startswith(("mingw", "msys", "cygwin")) else "linux")
    ram_gb = max(1, round(mem_b / (1024 ** 3))) if mem_b else (max(1, round(mem_kb / (1024 ** 2))) if mem_kb else None)

    if os_name == "macos" and arch == "arm64":
        hw_raw = f"{chip} ({gpu_cores}-core GPU)" if (chip and gpu_cores) else (chip or "Apple Silicon")
        return {
            "kind": "unified",
            "hw_id": map_apple_id(chip, gpu_cores),
            "hw_raw": hw_raw,
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": ram_gb,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": "macos",
        }

    gpus: List[Tuple[str, int]] = []
    for nv in nv_lines:
        parts = [p.strip() for p in nv.split(",")]
        if len(parts) >= 2:
            try:
                vram = max(1, round(float(parts[1]) / 1024))
            except ValueError:
                vram = 0
            gpus.append((parts[0], vram))
    if gpus:
        name, vram = gpus[0]
        if "gb10" in name.lower():
            return {
                "kind": "unified",
                "hw_id": "dgxspark",
                "hw_raw": name,
                "gpu_count": len(gpus),
                "vram_gb": None,
                "ram_gb": ram_gb or 128,
                "ram_type": "Unified",
                "ram_channels": None,
                "os": os_name,
            }
        return {
            "kind": "gpu",
            "hw_id": map_gpu_id(name, vram),
            "hw_raw": name,
            "gpu_count": len(gpus),
            "vram_gb": vram or None,
            "ram_gb": ram_gb,
            "ram_type": None,
            "ram_channels": None,
            "os": os_name,
        }
    return None


def is_local_endpoint(endpoint: str) -> bool:
    """Return True if endpoint points to the local machine (loopback)."""
    try:
        host = (urllib.parse.urlparse(endpoint).hostname or "").lower().strip("[]")
    except Exception:
        return False
    if host in ("localhost", "localhost.localdomain", "::1", "0.0.0.0"):
        return True
    if host.startswith("127."):
        return True
    return False

COMMON_ENDPOINTS = [
    "http://127.0.0.1:8080/v1",   # llama-server / Strata / mlx-serve
    "http://127.0.0.1:11434/v1",  # Ollama
    "http://127.0.0.1:1234/v1",   # LM Studio
    "http://127.0.0.1:8000/v1",   # vLLM / SGLang
    "http://127.0.0.1:30000/v1",  # SGLang default
    "http://127.0.0.1:8081/v1",   # alternate local port
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
    gsq = re.search(r"GSQ-RCO[\s_-]+(IQ[1-4]_[A-Z]+|Q[2-8]_[01K][A-Z_]*)", s, re.IGNORECASE)
    if gsq:
        return f"GSQ-RCO {gsq.group(1).upper()}"
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
    return val.lower() if val.lower().endswith("bit") else val.upper()


def detect_ram_gb() -> Optional[int]:
    sys_name = normalize_os()
    if sys_name == "macos":
        out = run_cmd(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return max(1, round(int(out) / (1024 ** 3)))
    elif sys_name == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return max(1, round(kb / (1024 ** 2)))
        except Exception:
            pass
    elif sys_name == "windows":
        out = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if out.isdigit():
            return max(1, round(int(out) / (1024 ** 3)))
    return None


def detect_hardware() -> Dict[str, Any]:
    """Collect hardware specs ONLY (no hostname, username, UUID, serial, or paths)."""
    os_name = normalize_os()
    ram_gb = detect_ram_gb()

    # 1. Apple Silicon (unified memory)
    if os_name == "macos" and platform.machine() == "arm64":
        chip = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        disp = run_cmd(["system_profiler", "SPDisplaysDataType"])
        cores_match = re.search(r"Total Number of Cores:\s*(\d+)", disp)
        gpu_cores = int(cores_match.group(1)) if cores_match else None
        hw_raw = f"{chip} ({gpu_cores}-core GPU)" if gpu_cores else chip
        return {
            "kind": "unified",
            "hw_id": map_apple_id(chip, gpu_cores),
            "hw_raw": hw_raw,
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": ram_gb,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": os_name,
        }

    # 2. NVIDIA GPU via nvidia-smi
    nv_out = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if nv_out:
        gpus: List[Tuple[str, int]] = []
        for line in nv_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    vram = max(1, round(float(parts[1]) / 1024))
                except ValueError:
                    vram = 0
                gpus.append((parts[0], vram))
        if gpus:
            name, vram = gpus[0]
            # Check for DGX Spark (GB10 unified memory)
            if "gb10" in name.lower():
                return {
                    "kind": "unified",
                    "hw_id": "dgxspark",
                    "hw_raw": name,
                    "gpu_count": len(gpus),
                    "vram_gb": None,
                    "ram_gb": ram_gb or 128,
                    "ram_type": "Unified",
                    "ram_channels": None,
                    "os": os_name,
                }
            return {
                "kind": "gpu",
                "hw_id": map_gpu_id(name, vram),
                "hw_raw": name,
                "gpu_count": len(gpus),
                "vram_gb": vram or None,
                "ram_gb": ram_gb,
                "ram_type": None,
                "ram_channels": None,
                "os": os_name,
            }

    # 3. AMD Ryzen AI Max+ 395 (unified) on Linux/Windows
    cpu_brand = ""
    if os_name == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_brand = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif os_name == "windows":
        cpu_brand = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ])
    if "ai max" in cpu_brand.lower() and "395" in cpu_brand.lower():
        return {
            "kind": "unified",
            "hw_id": "aimax395",
            "hw_raw": cpu_brand,
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": ram_gb,
            "ram_type": "Unified",
            "ram_channels": None,
            "os": os_name,
        }

    # 4. AMD / Intel discrete GPU fallback (Windows PowerShell or rocm-smi)
    if os_name == "windows":
        ps = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }",
        ])
        cands = []
        for line in ps.splitlines():
            if "|" not in line:
                continue
            name, ram_b = line.split("|", 1)
            name = name.strip()
            if not name or "microsoft basic" in name.lower():
                continue
            vram = max(1, round(int(ram_b) / (1024 ** 3))) if ram_b.strip().isdigit() and int(ram_b) > 0 else None
            cands.append((name, vram))
        if cands:
            name, vram = cands[0]
            return {
                "kind": "gpu",
                "hw_id": map_gpu_id(name, vram),
                "hw_raw": name,
                "gpu_count": sum(1 for c in cands if c[0] == name),
                "vram_gb": vram,
                "ram_gb": ram_gb,
                "ram_type": None,
                "ram_channels": None,
                "os": os_name,
            }
    elif os_name == "linux":
        rocm = run_cmd(["rocm-smi", "--showproductname", "--csv"])
        if rocm:
            lines = [l for l in rocm.splitlines() if l.strip() and not l.lower().startswith("device")]
            if lines:
                cols = [c.strip() for c in lines[0].split(",")]
                name = cols[1] if len(cols) > 1 else cols[0]
                return {
                    "kind": "gpu",
                    "hw_id": map_gpu_id(name, None),
                    "hw_raw": name,
                    "gpu_count": len(lines),
                    "vram_gb": None,
                    "ram_gb": ram_gb,
                    "ram_type": None,
                    "ram_channels": None,
                    "os": os_name,
                }

    return {
        "kind": "gpu",
        "hw_id": None,
        "hw_raw": cpu_brand or "Unknown Hardware",
        "gpu_count": 1,
        "vram_gb": None,
        "ram_gb": ram_gb,
        "ram_type": None,
        "ram_channels": None,
        "os": os_name,
    }


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Any:
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


COMMON_PORTS = [8080, 11434, 1234, 8000, 30000, 8081]


def endpoint_candidates(raw: str) -> List[str]:
    """Generate ordered candidate base URLs from flexible user input.

    Accepts:
      - '192.168.31.88'                       (auto-probes common ports + /v1 or root)
      - '192.168.31.88:11434'                 (auto-adds http:// and tries /v1 & root)
      - 'http://192.168.31.88:8080/v1'        (uses as-is)
      - 'http://192.168.31.88:8080/v1/chat/completions' (strips /chat/completions)
      - 'https://llm.lan/openai/v1/'          (keeps custom prefix)
    """
    s = (raw or "").strip()
    if not s:
        return []
    if not re.match(r"^https?://", s, re.IGNORECASE):
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
            # First try reverse-proxy default port (80/443), then common local LLM ports
            add(f"{base}/v1")
            add(base)
            for p in COMMON_PORTS:
                add(f"{scheme}://{host_fmt}:{p}/v1")
                add(f"{scheme}://{host_fmt}:{p}")
    return cands


def resolve_endpoint(raw: str, api_key: Optional[str] = None, explicit_model: Optional[str] = None) -> Tuple[str, List[str]]:
    """Normalize and probe a user-supplied endpoint (flexible IP/port/path)."""
    cands = endpoint_candidates(raw)
    if not cands:
        raise RuntimeError("Empty endpoint.")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err: Optional[Exception] = None
    for cand in cands:
        try:
            data = http_json(f"{cand}/models", headers=headers, timeout=2.5)
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [m["id"] for m in data["data"] if isinstance(m, dict) and m.get("id")]
                if models:
                    return cand, [explicit_model] if explicit_model else models
                if explicit_model:
                    return cand, [explicit_model]
        except Exception as e:
            last_err = e
            continue
    if explicit_model:
        return cands[0], [explicit_model]
    raise RuntimeError(
        f"无法连接到 {raw}（已自动尝试 {', '.join(cands[:4])}{' 等' if len(cands) > 4 else ''}）。"
        f"请确认推理服务已启动且监听在 0.0.0.0（局域网可访问）。最后错误：{last_err}"
    )


def discover_endpoint(api_key: Optional[str] = None) -> Tuple[str, List[str]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for ep in COMMON_ENDPOINTS:
        for cand in (ep.rstrip("/"), ep.rstrip("/").removesuffix("/v1") if hasattr(str, "removesuffix") else re.sub(r"/v1$", "", ep.rstrip("/"))):
            try:
                data = http_json(f"{cand}/models", headers=headers, timeout=2.0)
                models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
                if models:
                    return cand, models
            except Exception:
                continue
    raise RuntimeError(
        "No local OpenAI-compatible server found on common ports (8080, 11434, 1234, 8000, 30000, 8081). "
        "Start your local LLM server first or pass --endpoint <IP或完整URL>"
    )


def fetch_models(endpoint: str, api_key: Optional[str] = None) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = http_json(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=5.0)
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


def build_prompt(target_tokens: int, nonce: str) -> str:
    # Unique nonce at the beginning prevents KV prefix-cache reuse between runs.
    header = (
        f"[Bench run {nonce}] Read the reference passage below, then write a clear, "
        f"detailed technical explanation of how memory bandwidth affects local LLM inference "
        f"without stopping early.\n\n"
    )
    # Approximate ~4 characters per token for English prose.
    target_chars = max(0, target_tokens * 4 - len(header))
    if target_chars <= 0:
        return header
    reps = (target_chars // len(FILLER_PARAGRAPH)) + 1
    body = (FILLER_PARAGRAPH * reps)[:target_chars]
    return header + body


def run_single_completion(
    endpoint: str,
    model_name: str,
    prompt: str,
    gen_tokens: int,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    chunk_tokens = 0
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    completion_tokens = int(
        timings.get("predicted_n")
        or usage.get("completion_tokens")
        or max(1, chunk_tokens)
    )

    # SSE decode speed: tokens after the first token divided by decode interval.
    sse_tg_tps = max(0.1, (max(1, completion_tokens - 1)) / decode_s) if completion_tokens > 1 else max(0.1, 1.0 / ttft_s)
    sse_pp_tps = max(0.1, prompt_tokens / ttft_s)

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
    }


def benchmark_context(
    endpoint: str,
    model_name: str,
    ctx_prompt: int,
    gen_tokens: int,
    runs: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for i in range(runs):
        nonce = f"{ctx_prompt}-{i}-{int(time.time() * 1000) % 100000}"
        prompt = build_prompt(ctx_prompt, nonce)
        r = run_single_completion(endpoint, model_name, prompt, gen_tokens, api_key=api_key)
        samples.append(r)
        print(
            f"  [ctx≈{ctx_prompt}] run {i + 1}/{runs}: "
            f"tg={r['tg_tps']} tok/s, pp={r['pp_tps']} tok/s, "
            f"ttft={r['ttft_ms']}ms ({r['source']})",
            file=sys.stderr,
        )

    return {
        "ctx_prompt": int(round(statistics.median(s["prompt_tokens"] for s in samples))),
        "gen_tokens": int(round(statistics.median(s["gen_tokens"] for s in samples))),
        "tg_tps": round(statistics.median(s["tg_tps"] for s in samples), 2),
        "pp_tps": round(statistics.median(s["pp_tps"] for s in samples), 2),
        "sse_tg_tps": round(statistics.median(s["sse_tg_tps"] for s in samples), 2),
        "ttft_ms": round(statistics.median(s["ttft_ms"] for s in samples), 1),
        "runs": runs,
        "source": "timings" if all(s["source"] == "timings" for s in samples) else "sse",
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standardized, privacy-first speed test for local LLMs (https://itbanana.com/llm-hardware)."
    )
    p.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1 (auto-detected if omitted)")
    p.add_argument("--api-key", help="Optional Bearer API key for the local server")
    p.add_argument("--model", help="Served model name for /v1/chat/completions (defaults to first model from /v1/models)")
    p.add_argument("--model-id", help="Canonical model ID (e.g. qwen38-flash-next, qwen38-27b)")
    p.add_argument("--quant", help="Quantization label (e.g. 'GSQ-RCO IQ3_S', 'Q4_K_M', 'FP8')")
    p.add_argument("--engine", help="Inference engine (e.g. strata, llamacpp, vllm, sglang, mlx, maya)")
    p.add_argument("--engine-version", help="Engine version (e.g. 0.2.15)")
    p.add_argument("--backend", help="Compute backend (e.g. cuda, rocm, vulkan, metal)")
    p.add_argument("--mtp", choices=["true", "false"], help="Whether MTP / speculative decoding is enabled")
    p.add_argument("--ctx-limit", type=int, help="Server configured context limit (e.g. 131072)")
    p.add_argument("--fit", choices=["gpu", "hybrid"], help="Whether model fits fully in VRAM (gpu) or spills to RAM (hybrid)")

    # Hardware overrides (required for LAN/remote endpoints; optional overrides locally)
    p.add_argument("--ssh", help="Optional SSH target (e.g. user@192.168.1.50) to auto-detect LAN server hardware (never uploaded)")
    p.add_argument("--hw-kind", choices=["gpu", "unified"], help="Override hardware kind (gpu or unified)")
    p.add_argument("--hw-id", help="Override canonical hardware ID (e.g. rtx4090-48)")
    p.add_argument("--hw-raw", help="Override raw hardware description")
    p.add_argument("--gpu-count", type=int, help="Override GPU count")
    p.add_argument("--vram-gb", type=int, help="Override per-GPU VRAM in GB")
    p.add_argument("--ram-gb", type=int, help="Override system RAM in GB")
    p.add_argument("--ram-type", help="System RAM type (e.g. DDR4, DDR5, DDR3, Unified)")
    p.add_argument("--ram-channels", type=int, help="System RAM channels (e.g. 2, 4, 8)")
    p.add_argument("--os", dest="target_os", choices=["linux", "windows", "macos"], help="Override target machine OS")

    # Benchmark parameters
    p.add_argument("--ctx", default=None, help="Comma-separated prompt token sizes to test (default: 1024)")
    p.add_argument("--gen-tokens", type=int, default=256, help="Tokens to generate per run (default: 256)")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per context size (default: 3)")
    p.add_argument("--no-warmup", action="store_true", help="Skip the initial warmup request")

    # Upload / interaction / privacy controls
    p.add_argument("--ui", action="store_true", help="Open the local browser web UI (default when run with no flags in a terminal)")
    p.add_argument("--cli", action="store_true", help="Force terminal CLI mode instead of opening the local web UI")
    p.add_argument("--ui-port", type=int, default=0, help="Port for local web UI (default: random free port on 127.0.0.1)")
    p.add_argument("--detect-only", action="store_true", help="Only print detected hardware JSON and exit")
    p.add_argument("--dry-run", action="store_true", help="Run benchmark and print JSON without uploading")
    p.add_argument("--interactive", "-i", action="store_true", help="Force interactive terminal CLI mode")
    p.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts and web UI (recommended for AI agents / scripts)")
    p.add_argument("--yes", "-y", action="store_true", help="Upload without interactive confirmation prompt (CLI mode)")
    p.add_argument("--submit-url", default=DEFAULT_SUBMIT_URL, help=f"Upload endpoint (default: {DEFAULT_SUBMIT_URL})")
    p.add_argument("--stats-url", default=DEFAULT_STATS_URL, help=f"Stats endpoint (default: {DEFAULT_STATS_URL})")
    return p.parse_args(argv)


def resolve_hardware(args: argparse.Namespace, remote: bool) -> Dict[str, Any]:
    """Build hardware spec. For remote/LAN endpoints, never mix in local machine hardware."""
    if remote:
        hw = {
            "kind": None,
            "hw_id": None,
            "hw_raw": None,
            "detected_by": "manual",
            "gpu_count": 1,
            "vram_gb": None,
            "ram_gb": None,
            "ram_type": None,
            "ram_channels": None,
            "os": None,
        }
    else:
        hw = detect_hardware()
        hw["detected_by"] = "manual" if (args.hw_id or args.hw_raw) else "local"

    if args.hw_kind:
        hw["kind"] = args.hw_kind
    if args.hw_id:
        hw["hw_id"] = args.hw_id.lower()
        if not args.hw_kind:
            hw["kind"] = "unified" if hw["hw_id"] in UNIFIED_IDS else "gpu"
        if not args.hw_raw and (remote or not hw.get("hw_raw")):
            hw["hw_raw"] = HW_ID_LABELS.get(hw["hw_id"], hw["hw_id"])
    if hw.get("kind") == "gpu" and hw.get("ram_type") == "Unified" and not args.ram_type:
        hw["ram_type"] = None
    if args.hw_raw:
        hw["hw_raw"] = args.hw_raw
        if not args.hw_id and remote:
            hw["hw_id"] = map_gpu_id(args.hw_raw, args.vram_gb) or map_apple_id(args.hw_raw, None)
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

    if remote:
        if not hw["hw_id"] and not hw["hw_raw"]:
            raise ValueError(
                "Remote/LAN endpoint detected: local hardware auto-detection is disabled so we never "
                "misreport this client machine's specs as the server's.\n"
                "Recommended: SSH into the inference machine and run speedtest.py there.\n"
                "Or explicitly pass the server's hardware flags, e.g.:\n"
                "  --hw-id rtx4090-48 --vram-gb 48 --ram-gb 64 --ram-type DDR4 --os linux"
            )
        if not hw["kind"]:
            hw["kind"] = "unified" if (hw["hw_id"] in UNIFIED_IDS) else "gpu"
        if not hw["hw_raw"]:
            hw["hw_raw"] = HW_ID_LABELS.get(hw["hw_id"], hw["hw_id"])
    return hw


def _ask(prompt_text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt_text}{suffix}: ").strip()
    except EOFError:
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
    if parsed.get("quant") and not args.quant:
        args.quant = parsed["quant"]
    if parsed.get("mtp") is not None and args.mtp is None:
        args.mtp = parsed["mtp"]


def prompt_remote_hardware(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Minimal one-line prompt for LAN/remote server specs (supports casual sentence or user@ip SSH)."""
    print("\n检测到是局域网/远程服务（已关闭本机硬件探测，避免填错成你手里这台电脑）。", file=sys.stderr)
    raw_in = _ask(
        "一句话描述推理机配置即可（例：4090 48G + 64G DDR4 Strata / 双 7900 XTX 128G / M5 Ultra 256G；或填 user@ip 自动读取）"
    )
    if not raw_in:
        return None
    # Detect SSH target like 'user@192.168.1.50'
    if re.match(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$", raw_in):
        detected = detect_hardware_via_ssh(raw_in)
        if detected:
            print(f"已自动识别远端硬件：{detected['hw_raw']} ×{detected['gpu_count']} (RAM {detected.get('ram_gb') or '?'}GB)", file=sys.stderr)
            return detected
        print("SSH 未能免密读取，改为一句话手动填：", file=sys.stderr)
        raw_in = _ask("一句话描述推理机配置（例：4090 48G + 64G DDR4）")
        if not raw_in:
            return None

    parsed = parse_human_hardware(raw_in)
    _apply_parsed(args, None, parsed)
    return None


def run_interactive_wizard(
    args: argparse.Namespace, hw: Dict[str, Any], models: List[str], endpoint: str, remote: bool
) -> Tuple[str, str]:
    """Ultra-light interactive confirmation: at most 1–2 keypresses to start benchmarking."""
    # 1. Pick model only if server exposes multiple models
    if args.model:
        model_name = args.model
    elif len(models) > 1:
        print("\n检测到多个已加载模型，请选择：", file=sys.stderr)
        for idx, m in enumerate(models, 1):
            print(f"  {idx}) {m}", file=sys.stderr)
        pick = _ask_int("输入编号", 1) or 1
        model_name = models[min(max(1, pick), len(models)) - 1]
    else:
        model_name = models[0]

    # Best-effort auto fill engine & quant so user doesn't have to type them
    if not args.engine:
        args.engine = infer_engine_auto(endpoint, model_name)
    guessed_q = args.quant or infer_quant(model_name)

    vram_part = f" {hw['vram_gb']}GB" if hw.get("vram_gb") and str(hw["vram_gb"]) not in str(hw.get("hw_raw", "")) else ""
    ram_part = f" + {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    ddr_part = f" {hw['ram_type']}" if hw.get("ram_type") else ""
    eng_part = f" | 引擎: {args.engine}" if args.engine else ""
    q_part = f" | 量化: {guessed_q}" if guessed_q else ""
    print(
        f"\n即将测速：{hw['hw_raw']}{vram_part} ×{hw['gpu_count']}{ram_part}{ddr_part} | 模型: {model_name}{q_part}{eng_part}",
        file=sys.stderr,
    )

    # For remote, user just gave a one-line description — start straight away without nagging!
    # For local, offer ONE optional single-line supplement (Enter starts immediately).
    if not remote:
        missing = []
        if not args.engine:
            missing.append("引擎如 Strata/llama.cpp")
        if hw.get("kind") == "gpu" and not hw.get("ram_type"):
            missing.append("内存如 DDR4/DDR5")
        if args.mtp is None:
            missing.append("开MTP/不开MTP")
        if missing:
            extra = _ask(f"可选补充（如 {'、'.join(missing)}；直接回车立即开始测速）")
            if extra:
                _apply_parsed(args, hw, parse_human_note(extra))

    return model_name, args.ctx or "1024"


def print_summary(payload: Dict[str, Any]) -> None:
    hw = payload["hardware"]
    sw = payload["software"]
    mdl = payload["model"]
    print("\n================ 测速成绩单 ================", file=sys.stderr)
    print(f"硬件: {hw['hw_raw']} ×{hw['gpu_count']} (RAM: {hw.get('ram_gb') or '?'}GB {hw.get('ram_type') or ''})".strip(), file=sys.stderr)
    print(f"模型: {mdl['model_raw']} | 量化: {mdl.get('quant') or '未填'} | 引擎: {sw.get('engine') or '未填'}", file=sys.stderr)
    for r in payload["results"]:
        print(
            f"  ctx≈{r['ctx_prompt']} tok -> 生成 {r['tg_tps']} tok/s | 读题 {r.get('pp_tps') or '-'} tok/s | "
            f"首字 {r.get('ttft_ms') or '-'} ms ({r['runs']}轮中位数, {r['source']})",
            file=sys.stderr,
        )
    print("============================================", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)

    if args.detect_only:
        hw = resolve_hardware(args, remote=False)
        print(json.dumps(hw, indent=2, ensure_ascii=False))
        return 0

    # Launch the local browser UI when explicitly asked (--ui) OR when a human runs `python3 speedtest.py` with no flags
    launch_ui = args.ui or (
        len(raw_argv) == 0
        and sys.stdin.isatty()
        and not args.cli
        and not args.non_interactive
        and not args.yes
        and not args.dry_run
        and not args.interactive
    )
    if launch_ui:
        import ui as ui_mod
        return ui_mod.serve_ui(
            sys.modules[__name__],
            submit_url=args.submit_url,
            stats_url=args.stats_url,
            port=args.ui_port,
        )

    interactive = args.interactive or (sys.stdin.isatty() and not args.yes and not args.non_interactive)

    print(
        "🔒 隐私承诺：100% 开源零依赖，绝不收集用户名/主机名/路径/IP/SSH 账号/API Key；"
        "上传前完整打印明文 JSON，欢迎让你的 AI 先检查源码。",
        file=sys.stderr,
    )

    # 1. Discover or confirm endpoint first (so we know whether it's local or LAN/remote)
    if args.endpoint:
        try:
            endpoint, models = resolve_endpoint(args.endpoint, api_key=args.api_key, explicit_model=args.model)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        try:
            endpoint, models = discover_endpoint(api_key=args.api_key)
        except RuntimeError as e:
            if not interactive:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            print(f"\n{e}", file=sys.stderr)
            entered = _ask("请输入推理机 IP 或接口地址（如 192.168.31.88 或 http://IP:端口/自定义路径，不用写 /v1 也行）")
            if not entered:
                return 1
            try:
                endpoint, models = resolve_endpoint(entered, api_key=args.api_key, explicit_model=args.model)
            except RuntimeError as e2:
                print(f"Error: {e2}", file=sys.stderr)
                return 1

    remote = not is_local_endpoint(endpoint)
    ssh_hw: Optional[Dict[str, Any]] = None
    if remote and args.ssh:
        ssh_hw = detect_hardware_via_ssh(args.ssh)
        if ssh_hw:
            print(f"已通过 SSH 自动识别远端硬件：{ssh_hw['hw_raw']} ×{ssh_hw['gpu_count']}", file=sys.stderr)
        else:
            print(f"Warning: 无法通过 SSH ({args.ssh}) 自动识别硬件。", file=sys.stderr)
    if remote and interactive and not ssh_hw and not args.hw_id and not args.hw_raw:
        ssh_hw = prompt_remote_hardware(args)

    if ssh_hw:
        hw = ssh_hw
        hw["detected_by"] = "manual" if (args.hw_id or args.hw_raw) else "ssh"
        if args.hw_kind:
            hw["kind"] = args.hw_kind
        if args.hw_id:
            hw["hw_id"] = args.hw_id.lower()
        if args.hw_raw:
            hw["hw_raw"] = args.hw_raw
        if args.gpu_count:
            hw["gpu_count"] = args.gpu_count
        if args.vram_gb:
            hw["vram_gb"] = args.vram_gb
        if args.ram_gb:
            hw["ram_gb"] = args.ram_gb
        if args.ram_type:
            hw["ram_type"] = args.ram_type
        if args.ram_channels:
            hw["ram_channels"] = args.ram_channels
        if args.target_os:
            hw["os"] = args.target_os
    else:
        try:
            hw = resolve_hardware(args, remote=remote)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    if not models and not args.model:
        print("Error: could not determine model name. Pass --model <name>.", file=sys.stderr)
        return 1

    if interactive:
        model_name, ctx_str = run_interactive_wizard(args, hw, models, endpoint, remote)
    else:
        model_name = args.model or models[0]
        ctx_str = args.ctx or "1024"
        if not args.engine:
            args.engine = infer_engine_auto(endpoint, model_name)

    print(f"\nEndpoint: {endpoint}", file=sys.stderr)
    print(f"Model:    {model_name}", file=sys.stderr)
    print(f"Hardware: {hw['hw_raw']} (hw_id={hw['hw_id'] or 'unmapped'}, x{hw['gpu_count']}, RAM={hw['ram_gb']}GB)", file=sys.stderr)

    if not args.no_warmup:
        print("Running warmup request...", file=sys.stderr)
        run_single_completion(endpoint, model_name, "Say OK.", 24, api_key=args.api_key)

    ctx_list = [int(x.strip()) for x in ctx_str.split(",") if x.strip()]
    results = [
        benchmark_context(endpoint, model_name, c, args.gen_tokens, args.runs, api_key=args.api_key)
        for c in ctx_list
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "client_version": CLIENT_VERSION,
        "hardware": hw,
        "software": {
            "engine": args.engine,
            "engine_version": args.engine_version,
            "backend": args.backend,
            "mtp": True if args.mtp == "true" else (False if args.mtp == "false" else None),
            "ctx_limit": args.ctx_limit,
        },
        "model": {
            "model_id": args.model_id or map_model_id(model_name),
            "model_raw": model_name,
            "quant": args.quant or infer_quant(model_name),
            "fit": args.fit,
        },
        "results": results,
    }

    print_summary(payload)
    print("\n=== Benchmark Payload (no personal data included) ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("\nDry run complete (not uploaded).", file=sys.stderr)
        return 0

    if not args.yes:
        try:
            ans = input(f"\nUpload this payload to {args.submit_url}? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Upload cancelled.", file=sys.stderr)
            return 0

    try:
        resp = http_json(args.submit_url, payload=payload, timeout=15.0)
        rec_id = resp.get("record_id")
        print(f"\n🎉 上传成功：{json.dumps(resp, ensure_ascii=False)}", file=sys.stderr)
        if rec_id:
            print(f"查看你的实测记录：https://itbanana.com/llm-hardware?highlight={rec_id}#live-speedtest", file=sys.stderr)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"\nUpload failed (HTTP {e.code}): {detail}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUpload failed: {e}", file=sys.stderr)
        return 1

    # Optionally fetch aggregate stats for this hardware + model
    hw_id = hw.get("hw_id")
    model_id = payload["model"].get("model_id")
    if hw_id and model_id and args.stats_url:
        try:
            q = urllib.parse.urlencode({"hw_id": hw_id, "model_id": model_id, "gpu_count": hw["gpu_count"]})
            st = http_json(f"{args.stats_url}?{q}", timeout=5.0)
            grp = st.get("group")
            if grp and grp.get("contributors"):
                print(
                    f"Community stats for {model_id} on {hw_id} x{hw['gpu_count']}: "
                    f"{grp['tg_min']}–{grp['tg_max']} tok/s (median {grp['tg_median']}, {grp['contributors']} contributors).",
                    file=sys.stderr,
                )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
