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
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

CLIENT_VERSION = "0.3.2"
USER_AGENT = f"llm-speedtest/{CLIENT_VERSION} (+https://github.com/oktothemoon/llm-speedtest)"
SCHEMA_VERSION = 1
DEFAULT_SUBMIT_URL = "https://itbanana.com/api/llm-speedtest/submit"
DEFAULT_STATS_URL = "https://itbanana.com/api/llm-speedtest/stats"

DEFAULT_CTX_TIERS = [1024, 32768, 65536, 131072, 262144]
DEFAULT_CTX_STR = "1k,32k,64k,128k,256k"


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
    """Ensure UTF-8 output and ANSI color support on Windows/macOS/Linux terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
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


def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def clean_public_text(val: Optional[str], max_chars: int) -> Optional[str]:
    if not val:
        return None
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(val))
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
    "http://127.0.0.1:11434/v1",  # Ollama
    "http://127.0.0.1:1234/v1",   # LM Studio
    "http://127.0.0.1:8000/v1",   # vLLM / SGLang
    "http://127.0.0.1:30000/v1",  # SGLang default
    "http://127.0.0.1:8081/v1",   # alternate local port
    "http://127.0.0.1:8082/v1",   # alternate local port
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

    # 4. AMD / Intel discrete GPU fallback (Windows PowerShell or Linux ROCm/lspci)
    if os_name == "windows":
        # Prefer 64-bit registry qwMemorySize (avoids Win32_VideoController 4GB uint32 cap)
        ps = run_cmd([
            "powershell", "-NoProfile", "-Command",
            (
                "$reg = Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
                "-ErrorAction SilentlyContinue | Where-Object { $_.DriverDesc }; "
                "if ($reg) { $reg | ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" } } "
                "else { Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" } }"
            ),
        ])
        cands = []
        skip_win = ("microsoft basic", "microsoft remote", "virtual", "parsec", "citrix", "vmware")
        for line in ps.splitlines():
            if "|" not in line:
                continue
            name, ram_b = line.split("|", 1)
            name = name.strip()
            if not name or any(sk in name.lower() for sk in skip_win):
                continue
            vram = max(1, round(int(ram_b) / (1024 ** 3))) if ram_b.strip().isdigit() and int(ram_b) > 0 else None
            cands.append((name, vram))
        if cands:
            # Prefer a GPU that maps to a known hw_id or has the largest VRAM
            cands.sort(key=lambda c: (1 if map_gpu_id(c[0], c[1]) else 0, c[1] or 0), reverse=True)
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
        amd_smi = run_cmd(["amd-smi", "static", "--asic", "--vram", "--csv"]).splitlines()
        rocm = run_cmd(["rocm-smi", "--showproductname", "--csv"]).splitlines()
        rocm_mem = run_cmd(["rocm-smi", "--showmeminfo", "vram", "--csv"]).splitlines()
        pci = [
            l for l in run_cmd(["lspci"]).splitlines()
            if re.search(r"VGA|3D|Display", l, re.IGNORECASE)
        ]
        vram_bytes: List[int] = []
        try:
            drm_dir = "/sys/class/drm"
            if os.path.isdir(drm_dir):
                for entry in sorted(os.listdir(drm_dir)):
                    if not re.fullmatch(r"card\d+", entry):
                        continue
                    p = os.path.join(drm_dir, entry, "device", "mem_info_vram_total")
                    if os.path.isfile(p):
                        with open(p, "r", encoding="utf-8") as f:
                            raw_b = f.read().strip()
                            if raw_b.isdigit():
                                vram_bytes.append(int(raw_b))
        except Exception:
            pass
        non_nv = parse_linux_non_nvidia_gpus(rocm, rocm_mem, amd_smi, pci, vram_bytes)
        if non_nv:
            name, vram = non_nv[0]
            return {
                "kind": "gpu",
                "hw_id": map_gpu_id(name, vram),
                "hw_raw": name,
                "gpu_count": len(non_nv),
                "vram_gb": vram,
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


COMMON_PORTS = [8080, 11434, 1234, 8000, 30000, 8081, 8082, 8001, 9000, 5000]


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


def resolve_endpoint(raw: str, api_key: Optional[str] = None, explicit_model: Optional[str] = None) -> Tuple[str, List[str]]:
    """Normalize and probe a user-supplied local endpoint (rejects non-local hosts)."""
    cands = endpoint_candidates(raw)
    if not cands:
        raise RuntimeError("接口地址不能为空。")
    if not is_local_endpoint(cands[0]):
        raise ValueError(LOCAL_ONLY_ERR)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err: Optional[Exception] = None
    for cand in cands:
        try:
            data = http_json(f"{cand}/models", headers=headers, timeout=2.5)
            models = _extract_model_ids(data)
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
        f"无法连接到本机接口 {raw}（已尝试 {', '.join(cands[:4])}{' 等' if len(cands) > 4 else ''}）。\n"
        f"请确认本机推理服务已启动并加载模型。最后错误：{last_err}"
    )


def discover_endpoint(api_key: Optional[str] = None) -> Tuple[str, List[str]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    auth_hint: Optional[str] = None
    for ep in COMMON_ENDPOINTS:
        for cand in (ep.rstrip("/"), ep.rstrip("/").removesuffix("/v1") if hasattr(str, "removesuffix") else re.sub(r"/v1$", "", ep.rstrip("/"))):
            try:
                data = http_json(f"{cand}/models", headers=headers, timeout=2.0)
                models = _extract_model_ids(data)
                if models:
                    return cand, models
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and not auth_hint:
                    auth_hint = f"在 {cand} 发现推理服务但需要认证（HTTP {e.code}），请传入 --api-key <你的密钥>。"
                continue
            except Exception:
                continue
    if auth_hint:
        raise RuntimeError(auth_hint)
    raise RuntimeError(
        "未在本机常见端口（8080 / 11434 / 1234 / 8000 / 30000 / 8081 / 8082 / 8001 / 9000 / 5000）发现运行中的 OpenAI 兼容推理服务。\n"
        "请确认推理服务已启动并加载模型，或直接输入你的自定义端口号（如 8088）或 --endpoint http://127.0.0.1:<端口>/v1"
    )


def fetch_models(endpoint: str, api_key: Optional[str] = None) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = http_json(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=5.0)
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


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

    # 2. vLLM / SGLang / llama.cpp /v1/models metadata
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
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    chunk_tokens = 0
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}

    with _urlopen(req, timeout=timeout) as resp:
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
    tier_runs = effective_runs_for_ctx(ctx_prompt, runs)
    req_timeout = timeout_for_ctx(ctx_prompt)
    ctx_lbl = format_ctx_label(ctx_prompt)
    print(
        _c(f"\n⚡ 开始测速 [{ctx_lbl}]（上下文 ≈{ctx_prompt} tokens，共 {tier_runs} 轮，每轮生成 {gen_tokens} tokens）：", "1;36"),
        file=sys.stderr,
    )
    for i in range(tier_runs):
        nonce = f"{ctx_prompt}-{i}-{int(time.time() * 1000) % 100000}"
        prompt = build_prompt(ctx_prompt, nonce)
        r = run_single_completion(endpoint, model_name, prompt, gen_tokens, api_key=api_key, timeout=req_timeout)
        samples.append(r)
        src_lbl = "引擎 timings" if r["source"] == "timings" else "SSE 流式"
        print(
            f"   第 {i + 1}/{tier_runs} 轮：生成 {_c(str(r['tg_tps']), '1;33')} tok/s | "
            f"读题 {r['pp_tps']} tok/s | 首字 {r['ttft_ms']} ms（{src_lbl}）",
            file=sys.stderr,
        )

    return {
        "ctx_prompt": int(round(statistics.median(s["prompt_tokens"] for s in samples))),
        "gen_tokens": int(round(statistics.median(s["gen_tokens"] for s in samples))),
        "tg_tps": round(statistics.median(s["tg_tps"] for s in samples), 2),
        "pp_tps": round(statistics.median(s["pp_tps"] for s in samples), 2),
        "sse_tg_tps": round(statistics.median(s["sse_tg_tps"] for s in samples), 2),
        "ttft_ms": round(statistics.median(s["ttft_ms"] for s in samples), 1),
        "runs": tier_runs,
        "source": "timings" if all(s["source"] == "timings" for s in samples) else "sse",
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="llm-speedtest：本机大模型标准化测速工具（https://itbanana.com/llm-speed）。"
    )
    p.add_argument("--endpoint", help="本机 OpenAI 兼容接口地址，例如 http://127.0.0.1:8080/v1 或端口号 8080（不填则自动扫描）")
    p.add_argument("--api-key", help="本机推理服务 API Key（选填，绝不上传）")
    p.add_argument("--model", help="指定要测试的模型名（默认取 /v1/models 返回的第一个模型）")
    p.add_argument("--model-id", help="标准模型 ID（例如 qwen38-flash-next、qwen38-27b）")
    p.add_argument("--quant", help="量化版本（例如 'GSQ-RCO IQ3_S'、'Q4_K_M'、'FP8'）")
    p.add_argument("--engine", help="推理引擎（strata / llamacpp / vllm / sglang / mlx / maya）")
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
    p.add_argument("--gen-tokens", type=int, default=256, help="每轮生成 token 数（默认 256）")
    p.add_argument("--runs", type=int, default=3, help="短上下文每档测速轮数，取中位数（默认 3；≥64k 自动 1 轮）")
    p.add_argument("--no-warmup", action="store_true", help="跳过预热请求")

    # Optional leaderboard identity & tuning note (voluntary public fields; anonymous if omitted)
    p.add_argument("--name", "--nickname", dest="nickname", help="选填：天梯图公开署名（≤20字，不含网址，留空匿名）")
    p.add_argument("--note", help="选填：一句话调优备注（≤120字，如 'Strata 0.2.15 + fa=on'）")

    # Mode / upload / interaction controls
    p.add_argument("--ui", action="store_true", help="直接启动本地 WebUI 可视化页面并打开浏览器")
    p.add_argument("--cli", action="store_true", help="直接进入终端命令行交互模式（不询问模式选择）")
    p.add_argument("--ui-port", type=int, default=0, help="本地 WebUI 监听端口（默认在 127.0.0.1 随机空闲端口）")
    p.add_argument("--detect-only", action="store_true", help="仅输出本机自动识别的硬件 JSON 并退出")
    p.add_argument("--dry-run", action="store_true", help="仅本地测速并打印明文 JSON，不上传")
    p.add_argument("--interactive", "-i", action="store_true", help="强制开启终端命令行交互模式")
    p.add_argument("--non-interactive", action="store_true", help="关闭所有交互提示（适合脚本与 AI Agent 调用）")
    p.add_argument("--yes", "-y", action="store_true", help="测速完成后跳过确认直接上传（命令行模式）")
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
    suffix = f" [默认 {default}]" if default else ""
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
    if parsed.get("quant") and not args.quant:
        args.quant = parsed["quant"]
    if parsed.get("kv_cache") and not args.kv_cache:
        args.kv_cache = parsed["kv_cache"]
    if parsed.get("mtp") is not None and args.mtp is None:
        args.mtp = parsed["mtp"]


def print_welcome_banner() -> None:
    print(_c("╭────────────────────────────────────────────────────────────╮", "36"), file=sys.stderr)
    print(_c(f"│  ⚡ llm-speedtest v{CLIENT_VERSION} · 本地大模型标准化测速工具       │", "1;36"), file=sys.stderr)
    print(_c("╰────────────────────────────────────────────────────────────╯", "36"), file=sys.stderr)
    print(
        "🔒 隐私承诺：100% 开源、零第三方依赖，仅采集本机硬件规格与推理速度，\n"
        "   绝不收集用户名、主机名、文件路径、原始 IP 或 API Key。\n"
        "   上传前会完整展示明文 JSON，欢迎随时把源码交给你的 AI 审计。",
        file=sys.stderr,
    )


def prompt_launch_mode() -> str:
    """Let interactive terminal users choose CLI or local WebUI."""
    print("\n请选择操作方式：", file=sys.stderr)
    print(f"  {_c('[1]', '1;36')} 终端命令行交互测速（推荐，本机或 SSH 终端直接跑）", file=sys.stderr)
    print(f"  {_c('[2]', '1;36')} 打开本地 WebUI 可视化页面（自动在浏览器打开）", file=sys.stderr)
    choice = _ask("请输入选项 [1/2]", "1").lower()
    if choice in ("2", "ui", "web", "webui", "w"):
        return "ui"
    return "cli"


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
        print("\n📦 检测到本机已加载多个模型，请选择要测速的模型：", file=sys.stderr)
        for idx, m in enumerate(models, 1):
            print(f"   {idx}) {m}", file=sys.stderr)
        pick = _ask_int("请输入模型编号", 1) or 1
        model_name = models[min(max(1, pick), len(models)) - 1]
    else:
        model_name = models[0]

    if not args.engine:
        args.engine = infer_engine_auto(endpoint, model_name)
    guessed_q = args.quant or infer_quant(model_name)
    ctx_str = args.ctx or DEFAULT_CTX_STR

    vram_part = f" {hw['vram_gb']}GB" if hw.get("vram_gb") and str(hw["vram_gb"]) not in str(hw.get("hw_raw", "")) else ""
    ram_part = f" + 内存 {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    ddr_part = f" {hw['ram_type']}" if hw.get("ram_type") else ""
    eng_part = f" | 引擎: {args.engine}" if args.engine else ""
    q_part = f" | 量化: {guessed_q}" if guessed_q else ""
    kv_part = f" | KV: {normalize_kv_cache(args.kv_cache)}" if args.kv_cache else ""
    lim_part = f"（服务上限 ≈{format_ctx_label(detected_ctx_limit)}，超限档位自动跳过）" if detected_ctx_limit else ""
    print(
        f"\n🎯 即将测速：{_c(hw['hw_raw'] + vram_part, '1')} ×{hw['gpu_count']}{ram_part}{ddr_part}"
        f" | 模型: {_c(model_name, '1')}{q_part}{eng_part}{kv_part}",
        file=sys.stderr,
    )
    print(f"📐 测速上下文档位：{ctx_str}{lim_part}", file=sys.stderr)

    missing = []
    if not args.engine:
        missing.append("引擎如 Strata/llama.cpp")
    if not args.kv_cache:
        missing.append("KV如 kv int8/q8_0/fp8/f16")
    if hw.get("kind") == "gpu" and not hw.get("ram_type"):
        missing.append("内存如 DDR4/DDR5 双通道")
    if args.mtp is None:
        missing.append("开MTP/不开MTP")
    if missing:
        extra = _ask(f"✏️  可选一句话补充（如 {'、'.join(missing)}；直接回车立即开始测速）")
        if extra:
            _apply_parsed(args, hw, parse_human_note(extra))

    return model_name, ctx_str


def print_summary(payload: Dict[str, Any]) -> None:
    hw = payload["hardware"]
    sw = payload["software"]
    mdl = payload["model"]
    ram_str = f"{hw.get('ram_gb') or '?'}GB" + (f" {hw['ram_type']}" if hw.get("ram_type") else "")
    vram_str = f" ({hw['vram_gb']}GB 显存)" if hw.get("vram_gb") else ""
    kv_str = f" | KV Cache：{sw['kv_cache']}" if sw.get("kv_cache") else ""
    print(_c("\n╭────────────────────── 📊 测速成绩单 ──────────────────────╮", "36"), file=sys.stderr)
    print(f"  🖥️  本机硬件：{hw['hw_raw']}{vram_str} ×{hw['gpu_count']} · 内存 {ram_str} · {hw.get('os') or ''}", file=sys.stderr)
    print(f"  📦 测试模型：{mdl['model_raw']} | 量化：{mdl.get('quant') or '未填'} | 引擎：{sw.get('engine') or '未填'}{kv_str}", file=sys.stderr)
    if payload.get("nickname") or payload.get("note"):
        meta_parts = []
        if payload.get("nickname"):
            meta_parts.append(f"署名：{payload['nickname']}")
        if payload.get("note"):
            meta_parts.append(f"备注：{payload['note']}")
        print(f"  🏷️  {' | '.join(meta_parts)}", file=sys.stderr)
    for r in payload["results"]:
        src_lbl = "引擎 timings" if r["source"] == "timings" else "SSE 流式"
        lbl = format_ctx_label(r["ctx_prompt"])
        print(
            f"  ⚡ [{lbl:>4}] ctx≈{r['ctx_prompt']} → 生成 {_c(str(r['tg_tps']) + ' tok/s', '1;33')} | "
            f"读题 {r.get('pp_tps') or '-'} tok/s | 首字 {r.get('ttft_ms') or '-'} ms（{r['runs']}轮, {src_lbl}）",
            file=sys.stderr,
        )
    print(_c("╰───────────────────────────────────────────────────────────╯", "36"), file=sys.stderr)


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

    is_king = bool(ranking and ranking.get("same_hw_rank") == 1)
    title = (
        "👑 新王登基！你的本机实测已登顶同款硬件榜首！ 👑"
        if is_king
        else ("🏆 恭喜上榜！实测成绩已载入速度天梯图 🏆" if ranking else "🎉 恭喜！你的本机实测已成功上榜 🎉")
    )
    badge = f"实测编号 #{rec_id} · 🔒 本机实采认证" if rec_id else "🔒 本机实采认证"
    cnt = f" ×{hw['gpu_count']}" if (hw.get("gpu_count") or 1) > 1 else ""
    ram = f" · {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    quant = f" · {mdl['quant']}" if mdl.get("quant") else ""
    kv = f" · KV {sw['kv_cache']}" if sw.get("kv_cache") else ""
    author = payload.get("nickname") or "匿名贡献者"

    print("", file=sys.stderr)
    print(_c("╔══════════════════════════════════════════════════════════════╗", "1;33"), file=sys.stderr)
    print(_c(f"║  ✨🎊  {title}  🎊✨", "1;33"), file=sys.stderr)
    print(_c("╠══════════════════════════════════════════════════════════════╣", "33"), file=sys.stderr)
    print(f"  🎖️  {_c(badge, '1;36')}", file=sys.stderr)
    print(f"  🖥️  硬件：{hw.get('hw_raw', 'Unknown')}{cnt}{ram}", file=sys.stderr)
    print(f"  📦 模型：{mdl.get('model_raw', '-')}{quant}{kv}", file=sys.stderr)
    if len(results) == 1:
        print(
            f"  🚀 成绩：生成 {_c(str(r0.get('tg_tps', '-')) + ' tok/s', '1;32')}  |  "
            f"读题 {r0.get('pp_tps') or '-'} tok/s  |  署名：{author}",
            file=sys.stderr,
        )
    else:
        print(f"  🚀 多档实测成绩（共 {len(results)} 档） · 署名：{author}", file=sys.stderr)
        for r in results:
            lbl = format_ctx_label(r.get("ctx_prompt") or 0)
            print(
                f"     [{lbl:>4}] 生成 {_c(str(r.get('tg_tps', '-')) + ' tok/s', '1;32')}  |  "
                f"读题 {r.get('pp_tps') or '-'} tok/s  |  首字 {r.get('ttft_ms') or '-'} ms",
                file=sys.stderr,
            )
    if payload.get("note"):
        print(f"  📝 备注：{payload['note']}", file=sys.stderr)

    if ranking:
        gap = float(ranking.get("gap_to_first") or 0)
        gap_txt = "👑 当前同卡第一！" if gap <= 0.05 else f"🔥 距同卡第一仅差 {gap:g} tok/s，调优后再冲一次！"
        vs_med = ranking.get("vs_median_pct")
        med_txt = (
            f"（较同卡中位 {ranking.get('same_hw_median')} tok/s {'+' if vs_med >= 0 else ''}{vs_med}%）"
            if vs_med is not None
            else ""
        )
        print(_c("╟──────────────────────────────────────────────────────────────╢", "33"), file=sys.stderr)
        print(
            f"  🏆 同款硬件排名：第 {_c(str(ranking.get('same_hw_rank')), '1;33')} / {ranking.get('same_hw_total')} 名{med_txt}",
            file=sys.stderr,
        )
        print(
            f"  🌍 全榜综合排名：第 {_c(str(ranking.get('overall_rank')), '1;36')} / {ranking.get('overall_total')} 名 · {gap_txt}",
            file=sys.stderr,
        )

    if resp.get("total_contributors"):
        print(
            f"  🤝 全站已汇聚 {resp.get('total_samples') or 1} 条真实测速，感谢第 {resp.get('total_contributors')} 位贡献者！",
            file=sys.stderr,
        )
    if stats and isinstance(stats.get("group"), dict) and stats["group"].get("contributors"):
        g = stats["group"]
        print(
            f"  📈 同配置社区区间：{g['tg_min']}–{g['tg_max']} tok/s（中位 {g['tg_median']} tok/s，共 {g['contributors']} 位同好）",
            file=sys.stderr,
        )

    ladder_url = (ranking or {}).get("ladder_url") or (
        f"https://itbanana.com/llm-speed?highlight={rec_id}" if rec_id else "https://itbanana.com/llm-speed"
    )
    print(_c("╟──────────────────────────────────────────────────────────────╢", "33"), file=sys.stderr)
    print(f"  🔗 直达天梯图高亮我的成绩：{_c(ladder_url, '1;4;36')}", file=sys.stderr)
    print(_c("╚══════════════════════════════════════════════════════════════╝", "1;33"), file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    configure_console()
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)

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

    # When run with no flags in an interactive terminal, let the user choose CLI or WebUI
    if (
        len(raw_argv) == 0
        and sys.stdin.isatty()
        and not args.cli
        and not args.non_interactive
        and not args.yes
        and not args.dry_run
        and not args.interactive
    ):
        if prompt_launch_mode() == "ui":
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
    ram_hint = f" · 内存 {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    print(
        f"\n🖥️  本机硬件（自动采集）：{_c(hw['hw_raw'] + vram_hint, '1')} ×{hw['gpu_count']}{ram_hint} · 系统 {hw.get('os') or ''}",
        file=sys.stderr,
    )

    # 2. Discover or resolve local-only endpoint
    if args.endpoint:
        try:
            endpoint, models = resolve_endpoint(args.endpoint, api_key=args.api_key, explicit_model=args.model)
        except ValueError as e:
            print(_c(f"\n❌ {e}", "1;31"), file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(_c(f"\n❌ {e}", "1;31"), file=sys.stderr)
            return 1
    else:
        print("🔍 正在扫描本机常见推理端口（8080 / 11434 / 1234 / 8000 / 30000 / 8081）...", file=sys.stderr)
        try:
            endpoint, models = discover_endpoint(api_key=args.api_key)
        except RuntimeError as e:
            if not interactive:
                print(_c(f"\n❌ {e}", "1;31"), file=sys.stderr)
                return 1
            print(f"\n⚠️  {e}", file=sys.stderr)
            while True:
                entered = _ask("请输入本机服务端口号或地址（如 8080 或 http://127.0.0.1:9000/v1；输入 q 退出）")
                if not entered or entered.lower() in ("q", "quit", "exit"):
                    print("👋 已退出测速。", file=sys.stderr)
                    return 1
                try:
                    endpoint, models = resolve_endpoint(entered, api_key=args.api_key, explicit_model=args.model)
                    break
                except ValueError as e_local:
                    print(_c(f"❌ {e_local}", "1;31"), file=sys.stderr)
                except RuntimeError as e2:
                    print(_c(f"❌ {e2}", "1;31"), file=sys.stderr)

    if not is_local_endpoint(endpoint):
        print(_c(f"\n❌ {LOCAL_ONLY_ERR}", "1;31"), file=sys.stderr)
        return 2

    detected_ctx_limit = detect_server_ctx_limit(endpoint, api_key=args.api_key)
    effective_ctx_limit = args.ctx_limit or detected_ctx_limit
    ctx_lim_hint = f" · 上下文上限 {effective_ctx_limit}" if effective_ctx_limit else ""
    print(f"✅ 已连接本机推理服务：{_c(endpoint, '36')}（发现 {len(models)} 个已加载模型{ctx_lim_hint}）", file=sys.stderr)

    if not models and not args.model:
        print(_c("❌ 未检测到已加载模型，请先加载模型或通过 --model <模型名> 指定。", "1;31"), file=sys.stderr)
        return 1

    if interactive:
        model_name, ctx_str = run_interactive_wizard(args, hw, models, endpoint, detected_ctx_limit=effective_ctx_limit)
    else:
        model_name = args.model or models[0]
        ctx_str = args.ctx or DEFAULT_CTX_STR
        if not args.engine:
            args.engine = infer_engine_auto(endpoint, model_name)

    if not args.no_warmup:
        print("\n🔥 正在发送预热请求（避免冷启动影响成绩）...", file=sys.stderr)
        run_single_completion(endpoint, model_name, "Say OK.", 24, api_key=args.api_key)

    ctx_list = parse_ctx_list(ctx_str)
    results: List[Dict[str, Any]] = []
    for c in ctx_list:
        if effective_ctx_limit and c + args.gen_tokens > effective_ctx_limit and (results or c > effective_ctx_limit):
            print(
                f"⏭️  跳过 [{format_ctx_label(c)}]（当前服务上下文上限 {effective_ctx_limit} tokens，不足以测 {c} + {args.gen_tokens}）",
                file=sys.stderr,
            )
            continue
        try:
            results.append(benchmark_context(endpoint, model_name, c, args.gen_tokens, args.runs, api_key=args.api_key))
        except Exception as e:
            if results:
                print(
                    _c(f"⚠️  [{format_ctx_label(c)}] 测速失败（可能超出当前上下文上限或显存不足：{e}），已保留已完成档位成绩并停止更高档位。", "33"),
                    file=sys.stderr,
                )
                break
            raise

    if not results:
        print(_c("❌ 所有上下文档位均未能完成测速，请检查服务端上下文上限设置。", "1;31"), file=sys.stderr)
        return 1

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "client_version": CLIENT_VERSION,
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
            "model_id": args.model_id or map_model_id(model_name),
            "model_raw": model_name,
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
        in_nick = clean_public_text(_ask("🏷️  天梯图公开署名（选填，≤20字，直接回车保持匿名）"), 20)
        if in_nick:
            payload["nickname"] = in_nick
        in_note = clean_public_text(_ask("📝 一句话调优备注（选填，≤120字，方便同卡抄作业，直接回车跳过）"), 120)
        if in_note:
            payload["note"] = in_note

    print("\n📋 即将上传的完整明文 JSON（零个人隐私，请核对）：")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.dry_run:
        print(_c("\n✅ 试运行完成（--dry-run 模式，未上传数据）。", "32"), file=sys.stderr)
        return 0

    if not args.yes:
        ans = _ask(f"\n🚀 确认将这份实测成绩上传至 itbanana.com 速度天梯图吗？[y/N]", "N").lower()
        if ans not in ("y", "yes"):
            print("👋 已取消上传，本次测速成绩仅保留在本地终端。", file=sys.stderr)
            return 0

    print("\n⏳ 正在将实测成绩刻入 itbanana.com 速度天梯图...", file=sys.stderr)
    try:
        resp = http_json(args.submit_url, payload=payload, timeout=15.0)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(_c(f"\n❌ 上传失败（HTTP {e.code}）：{detail}", "1;31"), file=sys.stderr)
        return 1
    except Exception as e:
        print(_c(f"\n❌ 上传失败：{e}", "1;31"), file=sys.stderr)
        return 1

    # Optionally fetch aggregate stats for this hardware + model
    stats: Optional[Dict[str, Any]] = None
    hw_id = hw.get("hw_id")
    model_id = payload["model"].get("model_id")
    if hw_id and model_id and args.stats_url:
        try:
            q = urllib.parse.urlencode({"hw_id": hw_id, "model_id": model_id, "gpu_count": hw["gpu_count"]})
            stats = http_json(f"{args.stats_url}?{q}", timeout=5.0)
        except Exception:
            pass

    print_upload_ceremony(payload, resp, stats=stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
