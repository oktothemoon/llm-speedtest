#!/usr/bin/env python3
"""Unit & end-to-end mock server tests for llm-speedtest."""

import http.server
import io
import json
import threading
import time
import unittest
from typing import Any, Dict, List
from unittest import mock

import speedtest


class MappingTests(unittest.TestCase):
    def test_map_gpu_id_modded_and_standard(self):
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 4090", 48), "rtx4090-48")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 4090", 24), "rtx4090")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 4080 SUPER", 32), "rtx4080s-32")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 4080 SUPER", 16), "rtx4080s")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 3080", 20), "rtx3080-20")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA GeForce RTX 2080 Ti", 22), "rtx2080ti-22")
        self.assertEqual(speedtest.map_gpu_id("Tesla V100-SXM2-32GB", 32), "v100-32")
        self.assertEqual(speedtest.map_gpu_id("Tesla V100-PCIE-16GB", 16), "v100-16")
        self.assertEqual(speedtest.map_gpu_id("AMD Radeon RX 7900 XTX", 24), "rx7900xtx")
        self.assertEqual(speedtest.map_gpu_id("AMD Radeon AI PRO R9700", 32), "r9700")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA RTX PRO 5000 Blackwell", 72), "rtxpro5000-72")
        self.assertEqual(speedtest.map_gpu_id("NVIDIA RTX PRO 5000 Blackwell", 48), "rtxpro5000")

    def test_map_apple_id(self):
        self.assertEqual(speedtest.map_apple_id("Apple M5 Ultra", 80), "m5ultra")
        self.assertEqual(speedtest.map_apple_id("Apple M5 Ultra", 64), "m5ultra64")
        self.assertEqual(speedtest.map_apple_id("Apple M5 Max", 40), "m5max40")
        self.assertEqual(speedtest.map_apple_id("Apple M5 Max", 32), "m5max32")
        self.assertEqual(speedtest.map_apple_id("Apple M3 Ultra", 60), "m3ultra")

    def test_model_and_quant_inference(self):
        self.assertEqual(speedtest.map_model_id("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"), "qwen38-flash-next")
        self.assertEqual(speedtest.infer_quant("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"), "GSQ-RCO IQ3_S")
        self.assertEqual(speedtest.infer_quant("ISTA-DASLab-Qwen3.8-Flash-Next-IQ3_XXS.gguf"), "GSQ-RCO IQ3_XXS")
        self.assertEqual(speedtest.infer_quant("Qwen3.8-Flash-Next-UD-IQ3_XXS.gguf"), "UD-IQ3_XXS")
        self.assertEqual(speedtest.map_model_id("Qwen3.8-27B-Q4_K_M.gguf"), "qwen38-27b")
        self.assertEqual(speedtest.infer_quant("Qwen3.8-27B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(speedtest.map_model_id("DeepSeek-V4.1-Flash-UD-IQ4_XS"), "ds-v41-flash")
        self.assertEqual(speedtest.infer_quant("DeepSeek-V4.1-Flash-UD-IQ4_XS"), "UD-IQ4_XS")

    def test_local_only_endpoint_guard(self):
        self.assertTrue(speedtest.is_local_endpoint("http://127.0.0.1:8080/v1"))
        self.assertTrue(speedtest.is_local_endpoint("http://localhost:11434/v1"))
        self.assertTrue(speedtest.is_local_endpoint("http://[::1]:8080/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://192.168.1.50:8080/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://10.0.0.8:8000/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://gpu-box.local:8080/v1"))

        # Any LAN / remote endpoint MUST be rejected with exit code 2
        rc = speedtest.main(["--endpoint", "http://192.168.1.50:8080/v1", "--non-interactive", "--dry-run"])
        self.assertEqual(rc, 2)

        with self.assertRaises(ValueError):
            speedtest.resolve_endpoint("192.168.1.50:8080")

    def test_parse_human_note(self):
        n1 = speedtest.parse_human_note("DDR4 双通道 Strata 开MTP kv int8")
        self.assertEqual(n1["ram_type"], "DDR4")
        self.assertEqual(n1["ram_channels"], 2)
        self.assertEqual(n1["engine"], "strata")
        self.assertEqual(n1["mtp"], "true")
        self.assertEqual(n1["kv_cache"], "int8")

        n2 = speedtest.parse_human_note("DDR5 四通道 不开MTP llama.cpp Q4_K_M fp8 kv")
        self.assertEqual(n2["ram_type"], "DDR5")
        self.assertEqual(n2["ram_channels"], 4)
        self.assertEqual(n2["engine"], "llamacpp")
        self.assertEqual(n2["quant"], "Q4_K_M")
        self.assertEqual(n2["mtp"], "false")
        self.assertEqual(n2["kv_cache"], "fp8")

    def test_ctx_tiers_and_kv_cache_helpers(self):
        self.assertEqual(
            speedtest.parse_ctx_list("1k,32k,64k,128k,256k"),
            [1024, 32768, 65536, 131072, 262144],
        )
        self.assertEqual(speedtest.parse_ctx_list(None), [1024, 32768, 65536, 131072, 262144])
        self.assertEqual(speedtest.format_ctx_label(1024), "1k")
        self.assertEqual(speedtest.format_ctx_label(65536), "64k")
        self.assertEqual(speedtest.format_ctx_label(262144), "256k")
        self.assertEqual(speedtest.effective_runs_for_ctx(1024, 3), 3)
        self.assertEqual(speedtest.effective_runs_for_ctx(32768, 3), 2)
        self.assertEqual(speedtest.effective_runs_for_ctx(65536, 3), 1)
        self.assertEqual(speedtest.effective_runs_for_ctx(262144, 3), 1)
        self.assertEqual(speedtest.normalize_kv_cache("FP16"), "f16")
        self.assertEqual(speedtest.normalize_kv_cache("int8"), "int8")
        self.assertEqual(speedtest.normalize_kv_cache("q8"), "q8_0")
        self.assertEqual(speedtest.normalize_kv_cache("q8_0/q4_0"), "q8_0/q4_0")
        # RAM snapping (e.g. 61GB usable Linux MemTotal -> 64GB physical)
        self.assertEqual(speedtest.snap_ram_gb(61), 64)
        self.assertEqual(speedtest.snap_ram_gb(31), 32)
        self.assertEqual(speedtest.snap_ram_gb(125), 128)
        self.assertEqual(speedtest.snap_ram_gb(48), 48)
        # 256k tier on a 256k (262144) server trims slightly instead of skipping
        self.assertEqual(speedtest.effective_ctx_target(262144, 256, 262144), 262144 - 384)
        self.assertIsNone(speedtest.effective_ctx_target(262144, 256, 131072))
        # ANSI escape sequences from arrow/delete keys are stripped from public notes
        self.assertEqual(speedtest.clean_public_text("测试 \x1b[D\x1b[3~ beta", 120), "测试 beta")

    def test_parse_linux_non_nvidia_gpus(self):
        rocm = [
            "device,Card Series,Card Model,Card Vendor,Card SKU",
            "card0,AMD Radeon AI PRO R9700,0x7550,Advanced Micro Devices,D70201",
        ]
        gpus = speedtest.parse_linux_non_nvidia_gpus(rocm, [], [], [], [34342961152])
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0][0], "AMD Radeon AI PRO R9700")
        self.assertEqual(gpus[0][1], 32)
        self.assertEqual(speedtest.map_gpu_id(*gpus[0]), "r9700")

        pci = [
            "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT] (rev c0)"
        ]
        gpus_pci = speedtest.parse_linux_non_nvidia_gpus([], [], [], pci, [34359738368])
        self.assertEqual(len(gpus_pci), 1)
        self.assertEqual(speedtest.map_gpu_id(*gpus_pci[0]), "r9700")

    def test_endpoint_candidates_local_inputs(self):
        # Bare port number -> 127.0.0.1:<port>
        self.assertEqual(speedtest.endpoint_candidates("8080"), ["http://127.0.0.1:8080/v1", "http://127.0.0.1:8080"])
        self.assertEqual(speedtest.endpoint_candidates(":11434"), ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"])

        # localhost:port without scheme or /v1
        c2 = speedtest.endpoint_candidates("localhost:11434")
        self.assertEqual(c2, ["http://localhost:11434/v1", "http://localhost:11434"])

        # Full chat/completions URL pasted from another app
        c3 = speedtest.endpoint_candidates("http://127.0.0.1:9000/v1/chat/completions")
        self.assertEqual(c3, ["http://127.0.0.1:9000/v1"])

    def test_prompt_launch_mode(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(speedtest.prompt_launch_mode(), "cli")
        with mock.patch("builtins.input", return_value="1"):
            self.assertEqual(speedtest.prompt_launch_mode(), "cli")
        with mock.patch("builtins.input", return_value="2"):
            self.assertEqual(speedtest.prompt_launch_mode(), "ui")
        with mock.patch("builtins.input", return_value="ui"):
            self.assertEqual(speedtest.prompt_launch_mode(), "ui")
        with mock.patch("builtins.input", return_value="q"):
            self.assertEqual(speedtest.prompt_launch_mode(), "quit")

    def test_welcome_banner_box_alignment(self):
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            speedtest.print_welcome_banner()
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 8)
        widths = [speedtest._display_width(ln) for ln in lines]
        self.assertEqual(len(set(widths)), 1)
        self.assertLessEqual(widths[0], 80)
        self.assertIn("ITBANANA.COM", buf.getvalue())
        self.assertIn("https://itbanana.com/llm-speed", buf.getvalue())

    def test_persistent_anonymous_client_id(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cid_path = os.path.join(td, "nested", "client_id")
            with mock.patch.dict(os.environ, {"LLM_SPEEDTEST_CLIENT_ID_FILE": cid_path}):
                cid1 = speedtest.get_or_create_client_id()
                self.assertIsNotNone(speedtest.clean_client_id(cid1))
                self.assertTrue(os.path.isfile(cid_path))
                cid2 = speedtest.get_or_create_client_id()
                self.assertEqual(cid1, cid2)

                # Corrupted file should regenerate cleanly
                with open(cid_path, "w", encoding="utf-8") as f:
                    f.write("not valid!!\n")
                cid3 = speedtest.get_or_create_client_id()
                self.assertIsNotNone(speedtest.clean_client_id(cid3))
                self.assertNotEqual(cid3, "not valid!!")

    def test_terminal_cards_never_overflow_border(self):
        payload = {
            "hardware": {
                "hw_raw": "AMD Radeon AI PRO R9700",
                "gpu_count": 1,
                "ram_gb": 64,
                "vram_gb": 32,
                "os": "linux",
            },
            "software": {"engine": "llama.cpp", "kv_cache": ""},
            "model": {"model_raw": "qwen3.8-flash-next-iq3_s", "quant": "IQ3_S"},
            "nickname": "一凡",
            "results": [
                {"ctx_prompt": 1024, "tg_tps": 118.7, "pp_tps": 915.0, "ttft_ms": 851.7, "runs": 3, "source": "timings"},
                {"ctx_prompt": 262144, "tg_tps": 88.1, "pp_tps": 2955.2, "ttft_ms": 88587.2, "runs": 3, "source": "timings"},
            ],
        }
        resp = {
            "record_id": 11,
            "total_samples": 15,
            "total_contributors": 3,
            "ranking": {
                "same_hw_rank": 1,
                "same_hw_total": 3,
                "same_hw_median": 114.6,
                "vs_median_pct": 4,
                "same_model_rank": 1,
                "same_model_total": 3,
                "gap_to_first": 0.0,
            },
        }
        stats = {"group": {"tg_min": 93.5, "tg_max": 114.6, "tg_median": 94.9, "contributors": 3}}

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            speedtest.print_upload_ceremony(payload, resp, stats)
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        border_widths = [
            speedtest._display_width(ln)
            for ln in lines
            if ln.startswith(("╔", "║", "╠", "╟", "╚"))
        ]
        self.assertTrue(border_widths)
        self.assertEqual(len(set(border_widths)), 1)
        box_w = border_widths[0]
        for ln in lines:
            self.assertLessEqual(speedtest._display_width(ln), box_w)

        buf2 = io.StringIO()
        with mock.patch("sys.stderr", buf2):
            speedtest.print_summary(payload)
        sum_lines = [ln for ln in buf2.getvalue().splitlines() if ln.strip()]
        top_w = speedtest._display_width(sum_lines[0])
        bot_w = speedtest._display_width(sum_lines[-1])
        self.assertEqual(top_w, bot_w)
        for ln in sum_lines[1:-1]:
            self.assertLess(speedtest._display_width(ln), top_w)

    def test_ceremony_ranking_edge_cases(self):
        base_payload = {
            "hardware": {"hw_raw": "NVIDIA GeForce RTX 5070 Ti", "gpu_count": 1, "ram_gb": 64, "ram_type": "DDR4"},
            "software": {"kv_cache": None},
            "model": {"model_raw": "iq3_s", "quant": "IQ3_S"},
            "results": [{"ctx_prompt": 1024, "tg_tps": 65.6, "pp_tps": 658.0, "ttft_ms": 1584.9, "runs": 3}],
        }

        def _render(ranking, stats=None):
            buf = io.StringIO()
            resp = {"record_id": 56, "total_samples": 59, "total_contributors": 5, "ranking": ranking}
            with mock.patch("sys.stderr", buf):
                speedtest.print_upload_ceremony(base_payload, resp, stats)
            return buf.getvalue()

        # 1. Rank 1/1 with gap > 0 (own historical record is faster): must NOT claim king or "距同配置第一"
        out_not_pb = _render(
            {
                "same_hw_rank": 1,
                "same_hw_total": 1,
                "same_hw_median": 71.6,
                "vs_median_pct": -8,
                "same_model_rank": 5,
                "same_model_total": 5,
                "gap_to_first": 6.0,
            },
            stats={"group": {"tg_min": 71.6, "tg_max": 71.6, "tg_median": 71.6, "contributors": 1}},
        )
        self.assertNotIn("新王登基", out_not_pb)
        self.assertNotIn("距同配置第一", out_not_pb)
        self.assertNotIn("较同配置中位", out_not_pb)
        self.assertIn("本次较个人历史最佳低 6 tok/s", out_not_pb)
        self.assertIn("参考基准：71.6 tok/s（共 1 位实测用户）", out_not_pb)
        self.assertNotIn("71.6–71.6", out_not_pb)
        # Duplicate model_raw / quant should not render as "iq3_s · IQ3_S"
        self.assertNotIn("iq3_s · IQ3_S", out_not_pb)

        # 2. Sole pioneer (1/1, gap == 0): pioneer banner, not king
        out_pioneer = _render({
            "same_hw_rank": 1,
            "same_hw_total": 1,
            "same_hw_median": 65.6,
            "vs_median_pct": 0,
            "same_model_rank": 3,
            "same_model_total": 5,
            "gap_to_first": 0.0,
        })
        self.assertIn("首位开荒", out_pioneer)
        self.assertNotIn("新王登基", out_pioneer)
        self.assertIn("同配置首个基准", out_pioneer)

        # 3. True king (1/3, gap == 0)
        out_king = _render({
            "same_hw_rank": 1,
            "same_hw_total": 3,
            "same_hw_median": 60.0,
            "vs_median_pct": 9,
            "same_model_rank": 2,
            "same_model_total": 5,
            "gap_to_first": 0.0,
        })
        self.assertIn("新王登基", out_king)
        self.assertIn("较同配置中位 60.0 tok/s +9%", out_king)
        self.assertIn("当前同配置榜首", out_king)

        # 4. Non-first rank (2/4, gap > 0): gap must appear on same-hw line, NOT same-model line
        out_runner = _render({
            "same_hw_rank": 2,
            "same_hw_total": 4,
            "same_hw_median": 68.0,
            "vs_median_pct": -4,
            "same_model_rank": 4,
            "same_model_total": 10,
            "gap_to_first": 4.5,
        })
        self.assertNotIn("新王登基", out_runner)
        same_hw_line = next(ln for ln in out_runner.splitlines() if "★ 同配置同模型速度排名" in ln)
        same_model_line = next(ln for ln in out_runner.splitlines() if "★ 同模型速度排名" in ln)
        self.assertIn("距同配置第一仅差 4.5 tok/s", same_hw_line)
        self.assertNotIn("距同配置第一", same_model_line)

    def test_ceremony_personal_best_api(self):
        """New backend: ranks by personal best and sends is_personal_best / vs_personal_best."""
        base_payload = {
            "hardware": {"hw_raw": "NVIDIA GeForce RTX 5070 Ti", "gpu_count": 1, "ram_gb": 64, "ram_type": "DDR4"},
            "software": {"kv_cache": None},
            "model": {"model_raw": "iq3_s", "quant": "IQ3_S"},
            "results": [{"ctx_prompt": 1024, "tg_tps": 65.6, "pp_tps": 658.0, "ttft_ms": 1584.9, "runs": 3}],
        }

        def _render(ranking, record_id=56):
            buf = io.StringIO()
            resp = {"record_id": record_id, "total_samples": 59, "total_contributors": 5, "ranking": ranking}
            with mock.patch("sys.stderr", buf):
                speedtest.print_upload_ceremony(base_payload, resp, None)
            return buf.getvalue()

        common = {"same_model_rank": 1, "same_model_total": 3, "gap_to_first": 0}

        # 1. Sole config, slower than own best (the reported 71.6 → 65.6 case): no pioneer banner
        out = _render({**common, "same_hw_rank": 1, "same_hw_total": 1, "same_hw_best": 71.6,
                       "same_hw_median": 71.6, "vs_median_pct": 0, "submitted_tg": 65.6,
                       "personal_best": 71.6, "personal_best_tg": 71.6, "is_personal_best": False,
                       "vs_personal_best": -6, "personal_best_record_id": 40,
                       "ladder_url": "https://itbanana.com/llm-speed?highlight=40"})
        self.assertNotIn("首位开荒", out)
        self.assertNotIn("同配置首个基准", out)
        self.assertNotIn("新王登基", out)
        self.assertIn("本次较个人历史最佳低 6 tok/s（最佳 71.6 tok/s），未刷新纪录", out)
        self.assertIn("查看本次成绩 #56（天梯展示你的个人最佳 #40）", out)

        # 2. Already #1 of several, slower run: no king banner, median phrased as personal best
        out = _render({**common, "same_hw_rank": 1, "same_hw_total": 2, "same_hw_best": 80,
                       "same_hw_median": 77.5, "vs_median_pct": 3, "submitted_tg": 78,
                       "personal_best": 80, "is_personal_best": False, "vs_personal_best": -2,
                       "personal_best_record_id": 3})
        self.assertNotIn("新王登基", out)
        self.assertNotIn("当前同配置榜首", out)
        self.assertIn("个人最佳较同配置中位 77.5 tok/s +3%", out)
        self.assertIn("本次较个人历史最佳低 2 tok/s", out)

        # 3. Runner-up, slower than own best: shows both the gap to first and the personal-best gap
        out = _render({**common, "same_hw_rank": 2, "same_hw_total": 2, "same_hw_best": 80,
                       "same_hw_median": 77.5, "vs_median_pct": -3, "gap_to_first": 5,
                       "submitted_tg": 70, "personal_best": 75, "is_personal_best": False,
                       "vs_personal_best": -5, "personal_best_record_id": 4})
        lines = out.splitlines()
        i = next(k for k, ln in enumerate(lines) if "★ 同配置同模型速度排名" in ln)
        block = lines[i]
        for ln in lines[i + 1:]:  # 过长时在卡片内折行，续行以 5 个空格缩进
            if not ln.startswith("     "):
                break
            block += ln.strip()
        self.assertIn("距同配置第一仅差 5 tok/s", block)
        self.assertIn("本次较个人历史最佳低 5 tok/s（最佳 75 tok/s）", block)

        # 4. New personal best that is #1 among several: king banner, no "未刷新"
        out = _render({**common, "same_hw_rank": 1, "same_hw_total": 2, "same_hw_best": 80,
                       "same_hw_median": 75.8, "vs_median_pct": 6, "submitted_tg": 80,
                       "personal_best": 80, "is_personal_best": True, "vs_personal_best": 0,
                       "personal_best_record_id": 56})
        self.assertIn("新王登基", out)
        self.assertIn("较同配置中位 75.8 tok/s +6%", out)
        self.assertNotIn("个人最佳较", out)
        self.assertNotIn("未刷新纪录", out)
        self.assertIn("高亮我的成绩：", out)

        # 5. First-ever upload on a config: pioneer banner
        out = _render({**common, "same_hw_rank": 1, "same_hw_total": 1, "same_hw_best": 65.6,
                       "same_hw_median": 65.6, "vs_median_pct": 0, "submitted_tg": 65.6,
                       "personal_best": 65.6, "is_personal_best": True, "vs_personal_best": 0,
                       "personal_best_record_id": 56})
        self.assertIn("首位开荒", out)
        self.assertIn("同配置首个基准", out)

        # 6. Box borders stay aligned with the longer not-PB line
        widths = {speedtest._display_width(ln) for ln in _render({**common, "same_hw_rank": 2, "same_hw_total": 9,
                  "same_hw_median": 77.5, "vs_median_pct": -3, "gap_to_first": 5, "submitted_tg": 70,
                  "personal_best": 75, "is_personal_best": False, "vs_personal_best": -5,
                  "personal_best_record_id": 4}).splitlines() if ln.startswith(("╔", "║", "╠", "╟", "╚"))}
        self.assertEqual(len(widths), 1)

    def test_ui_celebration_handles_personal_best_fields(self):
        import ui
        html = ui._render_html("tok", "https://example.com/submit")
        self.assertIn("is_personal_best", html)
        self.assertIn("vs_personal_best", html)
        self.assertIn("personal_best_record_id", html)
        self.assertIn("与个人最佳 #", html)

    def test_cpu_name_cleanup_and_ram_type_detection(self):
        self.assertEqual(
            speedtest.clean_cpu_name("AMD Ryzen 9 9950X 16-Core Processor"),
            "AMD Ryzen 9 9950X",
        )
        self.assertEqual(
            speedtest.clean_cpu_name("13th Gen Intel(R) Core(TM) i9-13900K"),
            "Intel Core i9-13900K",
        )
        self.assertEqual(
            speedtest.clean_cpu_name("Intel(R) Xeon(R) Gold 6430 CPU @ 2.10GHz"),
            "Intel Xeon Gold 6430",
        )
        self.assertEqual(
            speedtest.clean_cpu_name("AMD Ryzen 7 7840HS w/ Radeon 780M Graphics"),
            "AMD Ryzen 7 7840HS",
        )
        self.assertEqual(speedtest.clean_cpu_name("Apple M4 Max"), "Apple M4 Max")

        # SMBIOS memory type codes
        self.assertEqual(speedtest.parse_smbios_memory_type(26), "DDR4")
        self.assertEqual(speedtest.parse_smbios_memory_type(34), "DDR5")
        self.assertEqual(speedtest.parse_smbios_memory_type(35), "LPDDR5")
        self.assertEqual(speedtest.parse_smbios_memory_type(36), "LPDDR5X")
        self.assertEqual(speedtest.parse_smbios_memory_type(0, 0, 6000), "DDR5")
        self.assertEqual(speedtest.parse_smbios_memory_type(0, 0, 3200), "DDR4")

        # High-confidence CPU -> DDR inference
        self.assertEqual(speedtest.infer_ram_type_from_cpu("AMD Ryzen 9 9950X"), "DDR5")
        self.assertEqual(speedtest.infer_ram_type_from_cpu("AMD EPYC 9654"), "DDR5")
        self.assertEqual(speedtest.infer_ram_type_from_cpu("Intel Core Ultra 9 285K"), "DDR5")
        self.assertEqual(speedtest.infer_ram_type_from_cpu("AMD Ryzen 9 5950X"), "DDR4")
        self.assertEqual(speedtest.infer_ram_type_from_cpu("AMD EPYC 7763"), "DDR4")
        self.assertEqual(speedtest.infer_ram_type_from_cpu("Intel Core i9-10900K"), "DDR4")
        # Intel 12/13/14th Gen supports both DDR4 & DDR5 -> must stay None
        self.assertIsNone(speedtest.infer_ram_type_from_cpu("Intel Core i9-13900K"))
        self.assertIsNone(speedtest.infer_ram_type_from_cpu("Intel Core i7-14700K"))

        # dmidecode parsing
        dmi = "Memory Device\n\tSize: 32 GB\n\tType: DDR5\n\tSpeed: 6000 MT/s\n"
        self.assertEqual(speedtest._parse_dmidecode_ram_type(dmi), "DDR5")

        # Windows SMBIOS mock
        with mock.patch.object(speedtest, "run_cmd", return_value="34|0|6000|6000\n34|0|6000|6000"):
            self.assertEqual(speedtest.detect_ram_type("windows", "Intel Core i9-13900K"), "DDR5")
        # Linux non-root fallback to CPU platform
        with mock.patch.object(speedtest, "run_cmd", return_value=""):
            self.assertEqual(speedtest.detect_ram_type("linux", "AMD Ryzen 9 9950X"), "DDR5")
            self.assertIsNone(speedtest.detect_ram_type("linux", "Intel Core i9-13900K"))


class EndToEndMockTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile

        self._tmp_dir = tempfile.TemporaryDirectory()
        self._cid_env = mock.patch.dict(
            os.environ,
            {"LLM_SPEEDTEST_CLIENT_ID_FILE": os.path.join(self._tmp_dir.name, "client_id")},
        )
        self._cid_env.start()
        self.uploaded: List[Dict[str, Any]] = []
        self.user_agents: List[str] = []
        uploaded_ref = self.uploaded
        ua_ref = self.user_agents

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/props":
                    self._send_json(200, {"default_generation_settings": {"n_ctx": 270000}})
                elif self.path == "/v1/models":
                    self._send_json(200, {"data": [
                        {"id": "Qwen3.8-27B-Q4_K_M"},
                        {"id": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"},
                    ]})
                elif self.path.startswith("/api/llm-speedtest/stats"):
                    self._send_json(200, {
                        "ok": True,
                        "total_samples": 1,
                        "total_contributors": 1,
                        "group": {"contributors": 1, "tg_min": 170, "tg_median": 170, "tg_max": 170},
                    })
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    prompt_len = max(1, round(len(body["messages"][0]["content"]) / 5.6))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(20):
                        chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.002)
                    final = {
                        "choices": [],
                        "usage": {"prompt_tokens": prompt_len, "completion_tokens": 20},
                        "timings": {
                            "prompt_n": prompt_len,
                            "predicted_n": 20,
                            "prompt_per_second": 3800.0,
                            "predicted_per_second": 170.5,
                        },
                    }
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                elif self.path == "/api/llm-speedtest/submit":
                    ua_ref.append(self.headers.get("User-Agent") or "")
                    uploaded_ref.append(body)
                    rec_id = len(uploaded_ref)
                    self._send_json(200, {
                        "ok": True,
                        "batch_id": "test-batch",
                        "record_id": rec_id,
                        "accepted": len(body["results"]),
                        "total_samples": rec_id,
                        "total_contributors": rec_id,
                        "results": [{"id": rec_id, "status": "approved"}],
                        "ranking": {
                            "model_id": body["model"].get("model_id"),
                            "record_id": rec_id,
                            "same_hw_rank": 1,
                            "same_hw_total": 1,
                            "overall_rank": 1,
                            "overall_total": 1,
                            "same_hw_median": 170.5,
                            "same_hw_best": 170.5,
                            "gap_to_first": 0,
                            "vs_median_pct": 0,
                            "ladder_url": f"https://itbanana.com/llm-speed?highlight={rec_id}",
                        },
                    })
                else:
                    self._send_json(404, {"error": "not found"})

            def _send_json(self, status, data):
                raw = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._cid_env.stop()
        self._tmp_dir.cleanup()

    def test_local_requests_bypass_http_proxy(self):
        base = f"http://127.0.0.1:{self.port}"
        with mock.patch.dict("os.environ", {"http_proxy": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1"}):
            ep, models = speedtest.resolve_endpoint(str(self.port))
            self.assertEqual(ep, f"{base}/v1")
            self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", models)

    def test_interactive_api_key_prompt_on_401(self):
        # Mock discover_endpoint raising AuthRequiredError, then user enters API key directly
        with mock.patch.object(speedtest, "discover_endpoint", side_effect=speedtest.AuthRequiredError(f"http://127.0.0.1:{self.port}/v1", 401)), \
             mock.patch("builtins.input", side_effect=["--api-key sk-test", "1", ""]):
            rc = speedtest.main([
                "--ctx", "1k",
                "--gen-tokens", "20",
                "--runs", "1",
                "--no-warmup",
                "--interactive",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)

    def test_end_to_end_benchmark_and_upload(self):
        base = f"http://127.0.0.1:{self.port}"
        rc = speedtest.main([
            "--endpoint", f"{base}/v1",
            "--model", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
            "--hw-id", "rtx4090-48",
            "--hw-raw", "NVIDIA GeForce RTX 4090",
            "--gpu-count", "1",
            "--vram-gb", "48",
            "--ram-gb", "64",
            "--ram-type", "DDR4",
            "--ram-channels", "2",
            "--engine", "strata",
            "--kv-cache", "int8",
            "--mtp", "true",
            "--fit", "hybrid",
            "--name", "老曹",
            "--note", "Strata 0.2.15 + fa=on",
            "--ctx", "1k,32k,64k,128k,256k",
            "--gen-tokens", "20",
            "--runs", "1",
            "--no-warmup",
            "--yes",
            "--submit-url", f"{base}/api/llm-speedtest/submit",
            "--stats-url", f"{base}/api/llm-speedtest/stats",
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.uploaded), 1)
        p = self.uploaded[0]
        self.assertEqual(p["schema_version"], 1)
        self.assertEqual(p["hardware"]["hw_id"], "rtx4090-48")
        self.assertEqual(p["software"]["kv_cache"], "int8")
        self.assertEqual(p["software"]["ctx_limit"], 270000)
        self.assertEqual(p["model"]["model_id"], "qwen38-flash-next")
        self.assertEqual(p["model"]["quant"], "GSQ-RCO IQ3_S")
        self.assertEqual(len(p["results"]), 5)
        self.assertEqual([speedtest.format_ctx_label(r["ctx_prompt"]) for r in p["results"]], ["1k", "32k", "64k", "128k", "256k"])
        self.assertEqual(p["results"][0]["tg_tps"], 170.5)
        self.assertEqual(p["results"][0]["source"], "timings")
        self.assertEqual(p["nickname"], "老曹")
        self.assertEqual(p["note"], "Strata 0.2.15 + fa=on")
        self.assertTrue(self.user_agents and self.user_agents[0].startswith("llm-speedtest/"))

    def test_interactive_wizard_flow_and_ceremony(self):
        base = f"http://127.0.0.1:{self.port}"
        # Pick model #2, one optional note line with kv int8, nickname, tuning note, press Enter (default Y) to upload
        answers = iter(["2", "DDR4 双通道 开MTP kv int8", "老曹", "Strata 0.2.15 + fa=on", ""])
        err_buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=lambda _prompt="": next(answers)), \
             mock.patch("sys.stderr", err_buf):
            rc = speedtest.main([
                "--endpoint", f"{base}/v1",
                "--hw-id", "rtx4090-48",
                "--hw-raw", "NVIDIA GeForce RTX 4090",
                "--vram-gb", "48",
                "--ram-gb", "64",
                "--ctx", "1k,64k",
                "--ctx-limit", "70000",
                "--gen-tokens", "20",
                "--runs", "2",
                "--no-warmup",
                "--interactive",
                "--submit-url", f"{base}/api/llm-speedtest/submit",
                "--stats-url", f"{base}/api/llm-speedtest/stats",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.uploaded), 1)
        p = self.uploaded[0]
        self.assertEqual(p["model"]["model_id"], "qwen38-flash-next")
        self.assertEqual(p["model"]["quant"], "GSQ-RCO IQ3_S")
        self.assertEqual(p["software"]["engine"], "strata")
        self.assertEqual(p["software"]["kv_cache"], "int8")
        self.assertEqual(p["software"]["mtp"], True)
        self.assertEqual(p["hardware"]["ram_type"], "DDR4")
        self.assertEqual(p["hardware"]["ram_channels"], 2)
        self.assertEqual(p["nickname"], "老曹")
        self.assertEqual(p["note"], "Strata 0.2.15 + fa=on")
        self.assertIsNotNone(speedtest.clean_client_id(p.get("client_id")))
        self.assertEqual(len(p["results"]), 2)
        err_text = err_buf.getvalue()
        self.assertIn("首位开荒", err_text)
        self.assertIn("同配置同模型速度排名", err_text)
        self.assertIn("同模型速度排名", err_text)
        self.assertIn("位实测用户", err_text)
        self.assertNotIn("同好", err_text)
        self.assertIn("实测编号 #1", err_text)
        self.assertIn("https://itbanana.com/llm-speed?highlight=1", err_text)

    def test_anonymous_and_custom_client_id_flags(self):
        base = f"http://127.0.0.1:{self.port}"
        rc1 = speedtest.main([
            "--endpoint", f"{base}/v1",
            "--model", "Qwen3.8-27B-Q4_K_M",
            "--hw-id", "rtx4090-48",
            "--hw-raw", "NVIDIA GeForce RTX 4090",
            "--ctx", "1k",
            "--gen-tokens", "20",
            "--runs", "1",
            "--no-warmup",
            "--non-interactive",
            "--yes",
            "--anonymous",
            "--submit-url", f"{base}/api/llm-speedtest/submit",
            "--stats-url", f"{base}/api/llm-speedtest/stats",
        ])
        self.assertEqual(rc1, 0)
        self.assertNotIn("client_id", self.uploaded[-1])

        custom_cid = "11111111-2222-4333-8444-555555555555"
        rc2 = speedtest.main([
            "--endpoint", f"{base}/v1",
            "--model", "Qwen3.8-27B-Q4_K_M",
            "--hw-id", "rtx4090-48",
            "--hw-raw", "NVIDIA GeForce RTX 4090",
            "--ctx", "1k",
            "--gen-tokens", "20",
            "--runs", "1",
            "--no-warmup",
            "--non-interactive",
            "--yes",
            "--client-id", custom_cid,
            "--submit-url", f"{base}/api/llm-speedtest/submit",
            "--stats-url", f"{base}/api/llm-speedtest/stats",
        ])
        self.assertEqual(rc2, 0)
        self.assertEqual(self.uploaded[-1].get("client_id"), custom_cid)

    def test_local_web_ui_end_to_end(self):
        import urllib.error
        import urllib.request
        import ui

        base = f"http://127.0.0.1:{self.port}"
        ui_srv, token = ui.create_ui_server(
            speedtest,
            submit_url=f"{base}/api/llm-speedtest/submit",
            stats_url=f"{base}/api/llm-speedtest/stats",
            port=0,
            gen_tokens_default=20,
            no_warmup=True,
        )
        ui_port = ui_srv.server_address[1]
        ui_base = f"http://127.0.0.1:{ui_port}"
        t = threading.Thread(target=ui_srv.serve_forever, daemon=True)
        t.start()
        try:
            # 1. GET / serves local-only HTML containing session token (no LAN/SSH tabs)
            with urllib.request.urlopen(f"{ui_base}/") as resp:
                html = resp.read().decode("utf-8")
                self.assertIn("llm-speedtest", html)
                self.assertIn(token, html)
                self.assertIn("anonymous-toggle", html)
                self.assertIn("同配置同模型速度排名", html)
                self.assertIn("同模型速度排名", html)
                self.assertNotIn("tab-lan", html)
                self.assertNotIn("api/ssh-detect", html)

            # 2. POST without token is rejected (403)
            bad_req = urllib.request.Request(
                f"{ui_base}/api/probe",
                data=json.dumps({"endpoint": base}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req)
            self.assertEqual(ctx.exception.code, 403)

            hdrs = {"X-Speedtest-Token": token}

            # 3. Probing a LAN/remote IP is rejected (400)
            with self.assertRaises(urllib.error.HTTPError) as ctx_lan:
                speedtest.http_json(f"{ui_base}/api/probe", payload={"endpoint": "192.168.1.50:8080"}, headers=hdrs)
            self.assertEqual(ctx_lan.exception.code, 400)

            # 4. Probe local mock inference server using bare port number
            probed = speedtest.http_json(f"{ui_base}/api/probe", payload={"endpoint": str(self.port)}, headers=hdrs)
            self.assertEqual(probed["endpoint"], f"{base}/v1")
            self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", probed["models"])

            # 5. Start benchmark job -> server MUST lock hardware to local auto-detect
            speedtest.http_json(f"{ui_base}/api/start", payload={
                "endpoint": f"{base}/v1",
                "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
                "quant": "GSQ-RCO IQ3_S",
                "engine": "strata",
                "kv_cache": "int8",
                "mtp": True,
                "ctx": "1k,32k,64k",
                "runs": 1,
                "ram_type": "DDR4",
                "ram_channels": 2,
            }, headers=hdrs)

            st = {}
            for _ in range(50):
                st = speedtest.http_json(f"{ui_base}/api/status")
                if st["state"] in ("done", "error"):
                    break
                time.sleep(0.05)
            self.assertEqual(st["state"], "done")
            self.assertEqual(st["payload"]["hardware"]["detected_by"], "local")
            self.assertIsNotNone(speedtest.clean_client_id(st["payload"].get("client_id")))
            self.assertEqual(st["payload"]["software"]["kv_cache"], "int8")
            self.assertEqual(st["payload"]["software"]["ctx_limit"], 270000)
            self.assertEqual(len(st["payload"]["results"]), 3)
            self.assertEqual(st["payload"]["results"][0]["tg_tps"], 170.5)

            # 6. Upload through UI backend
            up = speedtest.http_json(f"{ui_base}/api/upload", payload={"payload": st["payload"]}, headers=hdrs)
            self.assertTrue(up["ok"])
            self.assertEqual(len(self.uploaded), 1)
            self.assertEqual(self.uploaded[0]["hardware"]["detected_by"], "local")

            # 7. Anonymous mode in UI omits client_id
            speedtest.http_json(f"{ui_base}/api/start", payload={
                "endpoint": f"{base}/v1",
                "model": "Qwen3.8-27B-Q4_K_M",
                "ctx": "1k",
                "runs": 1,
                "anonymous": True,
            }, headers=hdrs)
            st2 = {}
            for _ in range(50):
                st2 = speedtest.http_json(f"{ui_base}/api/status")
                if st2["state"] in ("done", "error"):
                    break
                time.sleep(0.05)
            self.assertEqual(st2["state"], "done")
            self.assertNotIn("client_id", st2["payload"])
        finally:
            ui_srv.shutdown()
            ui_srv.server_close()


class TestLocalListeningDiscovery(unittest.TestCase):
    def test_parse_windows_netstat_and_tasklist(self):
        tl_out = "\n".join([
            '"System Idle Process","0","Services","0","8 K"',
            '"svchost.exe","1148","Services","0","24,000 K"',
            '"custom-backend.exe","19284","Console","1","450,000 K"',
            '"strata-server.exe","20001","Console","1","120,000 K"',
            '"chrome.exe","30001","Console","1","200,000 K"',
        ])
        ns_out = "\n".join([
            "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1148",
            "  TCP    0.0.0.0:49664          0.0.0.0:0              LISTENING       1148",
            "  TCP    127.0.0.1:47831        0.0.0.0:0              LISTENING       19284",
            "  TCP    127.0.0.1:17880        0.0.0.0:0              LISTENING       20001",
            "  TCP    127.0.0.1:9222         0.0.0.0:0              LISTENING       30001",
            "  TCP    127.0.0.1:51234        0.0.0.0:0              LISTENING       99999",
        ])
        pm = speedtest._parse_windows_tasklist(tl_out)
        entries = speedtest._parse_windows_netstat(ns_out, pm)
        self.assertIn((1, "127.0.0.1", 17880), entries)
        # Arbitrary process name or unknown PID on 127.0.0.1 is still discovered (Tier 3)
        self.assertIn((3, "127.0.0.1", 47831), entries)
        self.assertIn((3, "127.0.0.1", 51234), entries)
        # svchost and chrome must be skipped
        self.assertFalse(any(p in (135, 49664, 9222) for _, _, p in entries))

    def test_parse_windows_listening(self):
        ps_out = "\n".join([
            "127.0.0.1|17880|strata-server",
            "::1|41234|Strata",
            "0.0.0.0|135|svchost",
            "127.0.0.1|54321|python",
            "0.0.0.0|80|System",
        ])
        entries = speedtest._parse_windows_listening(ps_out)
        self.assertIn((1, "127.0.0.1", 17880), entries)
        self.assertIn((1, "[::1]", 41234), entries)
        self.assertIn((2, "127.0.0.1", 54321), entries)
        # svchost and port 80 must be filtered out
        self.assertFalse(any(p == 135 or p == 80 for _, _, p in entries))

    def test_parse_macos_and_linux_listening(self):
        lsof_out = "\n".join([
            "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME",
            "strata  12345 user   10u  IPv4 0x1234      0t0  TCP 127.0.0.1:17881 (LISTEN)",
            "Chrome  12346 user   11u  IPv4 0x1235      0t0  TCP 127.0.0.1:9222 (LISTEN)",
        ])
        mac_entries = speedtest._parse_macos_listening(lsof_out)
        self.assertEqual(mac_entries, [(1, "127.0.0.1", 17881)])

        ss_out = "\n".join([
            "State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process",
            'LISTEN 0 128 127.0.0.1:19999 0.0.0.0:* users:(("llama-server",pid=999,fd=5))',
            'LISTEN 0 128 127.0.0.1:22 0.0.0.0:* users:(("sshd",pid=100,fd=3))',
        ])
        lin_entries = speedtest._parse_linux_listening(ss_out)
        self.assertEqual(lin_entries, [(1, "127.0.0.1", 19999)])

    def test_discover_endpoint_dynamic_fallback_and_empty_models(self):
        with mock.patch.object(speedtest, "COMMON_ENDPOINTS", []), \
             mock.patch.object(speedtest, "_discover_listening_endpoints", return_value=["http://127.0.0.1:41234/v1"]), \
             mock.patch.object(speedtest, "http_json", return_value={"data": [{"id": "Qwen3.5-27B"}]}):
            ep, models = speedtest.discover_endpoint()
            self.assertEqual(ep, "http://127.0.0.1:41234/v1")
            self.assertEqual(models, ["Qwen3.5-27B"])

        with mock.patch.object(speedtest, "COMMON_ENDPOINTS", []), \
             mock.patch.object(speedtest, "_discover_listening_endpoints", return_value=["http://127.0.0.1:17880/v1"]), \
             mock.patch.object(speedtest, "http_json", return_value={"data": []}):
            with self.assertRaises(RuntimeError) as ctx:
                speedtest.discover_endpoint()
            self.assertIn("未加载任何模型", str(ctx.exception))

    def test_detect_server_model_id_from_props(self):
        with mock.patch.object(
            speedtest,
            "http_json",
            return_value={"model_path": r"C:\models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf"},
        ):
            mid = speedtest.detect_server_model_id("http://127.0.0.1:47831/v1", "iq3_s")
            self.assertEqual(mid, "qwen38-flash-next")

    def test_infer_engine_extended_and_props_fallback(self):
        self.assertEqual(speedtest.infer_engine_from_text("ik_llama-server"), "llamacpp")
        self.assertEqual(speedtest.infer_engine_from_text("llama_server.exe"), "llamacpp")
        self.assertEqual(speedtest.infer_engine_from_text("koboldcpp"), "llamacpp")

        class _FakeResp:
            def __init__(self, payload: dict):
                self._raw = json.dumps(payload).encode("utf-8")
                self.headers = {}

            def read(self):
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # llama.cpp /props with opaque build_info + default_generation_settings -> llamacpp
        with mock.patch.object(
            speedtest,
            "_urlopen",
            return_value=_FakeResp({"build_info": "b4521-deadbeef", "default_generation_settings": {"n_ctx": 131072}}),
        ):
            self.assertEqual(speedtest._detect_http_engine("http://127.0.0.1:47831/v1"), "llamacpp")

        # /props with GSQ-RCO model_path -> strata
        with mock.patch.object(
            speedtest,
            "_urlopen",
            return_value=_FakeResp({
                "build_info": "b4521-deadbeef",
                "model_path": r"C:\Users\alice\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf",
                "default_generation_settings": {"n_ctx": 131072},
            }),
        ):
            self.assertEqual(speedtest._detect_http_engine("http://127.0.0.1:47831/v1"), "strata")

    def test_resolve_model_raw_privacy_and_fallback(self):
        # Recognizable API name stays untouched
        self.assertEqual(
            speedtest.resolve_model_raw("http://127.0.0.1:8080/v1", "Qwen3.8-Flash-Next-IQ3_S"),
            "Qwen3.8-Flash-Next-IQ3_S",
        )
        # Short alias -> pure filename stem from /props (never leaks directory path)
        with mock.patch.object(
            speedtest,
            "http_json",
            return_value={"model_path": r"C:\Users\secret\models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf"},
        ):
            resolved = speedtest.resolve_model_raw("http://127.0.0.1:47831/v1", "iq3_s", model_id="qwen38-flash-next")
            self.assertEqual(resolved, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S")
            self.assertNotIn("secret", resolved)
            self.assertNotIn("\\", resolved)
        # /props unavailable -> fall back to '<Label> (<alias>)'
        with mock.patch.object(speedtest, "http_json", side_effect=OSError("no props")):
            resolved = speedtest.resolve_model_raw("http://127.0.0.1:47831/v1", "iq3_s", model_id="qwen38-flash-next")
            self.assertEqual(resolved, "Qwen3.8-Flash-Next (iq3_s)")

    def test_build_warmup_prompt(self):
        p = speedtest.build_warmup_prompt(131072)
        self.assertGreater(len(p), 800)
        self.assertIn("[Bench run warmup]", p)
        p_small = speedtest.build_warmup_prompt(200)
        self.assertLess(len(p_small), len(p))


if __name__ == "__main__":
    unittest.main()


class BenchmarkRobustnessTests(unittest.TestCase):
    """Mock OpenAI-compatible servers covering early EOS, missing usage, strict params and errors."""

    def _serve(self, mode: str):
        seen: List[Dict[str, Any]] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                code, data = (200, {"data": [{"id": "Qwen3-8B-Q4_K_M"}]}) if self.path == "/v1/models" else (404, {})
                raw = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                seen.append(body)
                if mode == "err500":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error":"model not loaded"}')
                    return
                if mode == "strict" and "ignore_eos" in body:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"unknown field ignore_eos"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                n = body["max_tokens"]
                stop_at = n if body.get("ignore_eos") else min(n, 30)
                per_chunk = 4 if mode == "multi_chunk_no_usage" else 1
                sent = 0
                while sent < stop_at:
                    k = min(per_chunk, stop_at - sent)
                    chunk = {"choices": [{"delta": {"content": "tok " * k}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    sent += k
                    time.sleep(0.001 * k)
                if mode != "multi_chunk_no_usage":
                    usage = {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": sent}}
                    self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        speedtest._NO_IGNORE_EOS_ENDPOINTS.clear()
        return f"http://127.0.0.1:{srv.server_address[1]}/v1", seen

    def test_ignore_eos_is_sent_so_generation_runs_full_length(self):
        ep, seen = self._serve("early_eos")
        r = speedtest.run_single_completion(ep, "m", "hello", 128)
        self.assertTrue(seen[0].get("ignore_eos"))
        self.assertEqual(r["gen_tokens"], 128)
        self.assertFalse(r["tokens_estimated"])

    def test_strict_server_rejecting_ignore_eos_falls_back_once(self):
        ep, seen = self._serve("strict")
        r1 = speedtest.run_single_completion(ep, "m", "hello", 128)
        r2 = speedtest.run_single_completion(ep, "m", "hello", 128)
        self.assertEqual([("ignore_eos" in b) for b in seen], [True, False, False])
        self.assertEqual(r1["gen_tokens"], 30)
        self.assertEqual(r2["gen_tokens"], 30)
        warns = speedtest.tier_quality_warnings([r1, r2], 128)
        self.assertTrue(any("不会计入速度天梯" in w for w in warns))

    def test_multi_token_chunks_without_usage_are_not_undercounted(self):
        ep, _ = self._serve("multi_chunk_no_usage")
        r = speedtest.run_single_completion(ep, "m", "hello", 256)
        # 64 chunks × 4 tokens: counting chunks alone would report 64 tokens (4× too slow)
        self.assertEqual(r["gen_tokens"], 256)
        self.assertTrue(r["tokens_estimated"])
        warns = speedtest.tier_quality_warnings([r], 256)
        self.assertTrue(any("未返回生成 token 数" in w for w in warns))

    def test_no_warnings_for_normal_tier(self):
        ep, _ = self._serve("early_eos")
        r = speedtest.run_single_completion(ep, "m", "hello", 256)
        self.assertEqual(speedtest.tier_quality_warnings([r], 256), [])

    def test_cli_turns_server_error_into_short_message(self):
        ep, _ = self._serve("err500")
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf), mock.patch("sys.stdout", io.StringIO()):
            code = speedtest.cli(["--endpoint", ep, "--ctx", "1k", "--runs", "1", "--dry-run",
                                  "--non-interactive", "--hw-id", "rtx4090"])
        out = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("HTTP 500", out)
        self.assertIn("model not loaded", out)
        self.assertNotIn("Traceback", out)

    def test_cli_ctrl_c_exits_cleanly(self):
        buf = io.StringIO()
        with mock.patch.object(speedtest, "main", side_effect=KeyboardInterrupt), mock.patch("sys.stderr", buf):
            code = speedtest.cli([])
        self.assertEqual(code, 130)
        self.assertIn("已中断测速", buf.getvalue())

    def test_ui_server_rejects_foreign_host_header(self):
        import urllib.error
        import ui

        srv, tok = ui.create_ui_server(speedtest, "http://127.0.0.1:9/x", "")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        port = srv.server_address[1]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def _get(host):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"Host": host})
            try:
                with opener.open(req, timeout=5) as r:
                    return r.status, r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return e.code, ""

        code, body = _get(f"attacker.example:{port}")
        self.assertEqual(code, 403)
        self.assertNotIn(tok, body)
        code, body = _get(f"127.0.0.1:{port}")
        self.assertEqual(code, 200)
        self.assertIn(tok, body)
        self.assertEqual(_get(f"localhost:{port}")[0], 200)


class HardwareSelectionTests(unittest.TestCase):
    """GPU counting for mixed cards, NVIDIA + AMD boxes and stale Windows adapters."""

    def _detect(self, outputs: Dict[str, str], os_name: str = "linux"):
        def fake_run_cmd(cmd, timeout=5.0):
            return outputs.get(cmd[0], "")

        with mock.patch.object(speedtest, "run_cmd", side_effect=fake_run_cmd), \
                mock.patch.object(speedtest, "normalize_os", return_value=os_name), \
                mock.patch.object(speedtest, "detect_ram_gb", return_value=64), \
                mock.patch.object(speedtest, "detect_cpu_raw", return_value="AMD Ryzen 9 9950X"), \
                mock.patch.object(speedtest, "detect_ram_type", return_value="DDR5"), \
                mock.patch("os.path.isdir", return_value=False):
            hw = speedtest.detect_hardware()
        return hw, list(speedtest.LAST_DETECT_NOTES)

    def test_parse_nvidia_smi_handles_na_memory_used(self):
        gpus = speedtest.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 21000\nNVIDIA GeForce RTX 3060, 12288, [N/A]\n")
        self.assertEqual(gpus[0], {"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24, "used_mb": 21000})
        self.assertIsNone(gpus[1]["used_mb"])

    def test_mixed_models_are_not_counted_as_same_card(self):
        name, vram, count, others = speedtest.choose_gpu_group(
            [("NVIDIA GeForce RTX 3060", 12), ("NVIDIA GeForce RTX 4090", 24)]
        )
        self.assertEqual((name, vram, count), ("NVIDIA GeForce RTX 4090", 24, 1))
        self.assertEqual(others, ["NVIDIA GeForce RTX 3060"])

    def test_only_gpus_holding_the_model_are_counted(self):
        hw, notes = self._detect({"nvidia-smi": (
            "NVIDIA GeForce RTX 4090, 24564, 22000\n"
            "NVIDIA GeForce RTX 4090, 24564, 350\n"
        )})
        self.assertEqual(hw["gpu_count"], 1)
        self.assertTrue(any("1 张已加载模型" in n for n in notes))

    def test_two_identical_busy_gpus_count_as_two(self):
        hw, notes = self._detect({"nvidia-smi": (
            "NVIDIA GeForce RTX 4090, 24564, 20000\n"
            "NVIDIA GeForce RTX 4090, 24564, 19000\n"
        )})
        self.assertEqual(hw["gpu_count"], 2)
        self.assertEqual(notes, [])

    def test_idle_nvidia_plus_amd_dgpu_reports_the_amd_card(self):
        hw, notes = self._detect({
            "nvidia-smi": "NVIDIA GeForce RTX 5070 Ti, 16303, 420\n",
            "lspci": "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon AI PRO R9700] (rev c0)\n",
            "rocm-smi": "",
        })
        self.assertIn("R9700", hw["hw_raw"])
        self.assertTrue(any("未发现已加载的模型" in n for n in notes))

    def test_busy_nvidia_plus_amd_dgpu_reports_nvidia_with_note(self):
        hw, notes = self._detect({
            "nvidia-smi": "NVIDIA GeForce RTX 5070 Ti, 16303, 14000\n",
            "lspci": "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon AI PRO R9700] (rev c0)\n",
        })
        self.assertIn("5070 Ti", hw["hw_raw"])
        self.assertTrue(any("非 NVIDIA" in n for n in notes))

    def test_unknown_nvidia_usage_keeps_previous_behaviour(self):
        hw, _ = self._detect({
            "nvidia-smi": "NVIDIA GeForce RTX 5070 Ti, 16303, [N/A]\n",
            "lspci": "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon AI PRO R9700] (rev c0)\n",
        })
        self.assertIn("5070 Ti", hw["hw_raw"])

    def test_windows_lines_skip_nvidia_and_virtual_adapters(self):
        cands = speedtest.parse_windows_gpu_lines(
            "NVIDIA GeForce RTX 4090|25757220864\n"
            "Microsoft Basic Display Adapter|0\n"
            "AMD Radeon RX 7900 XTX|25753026560\n"
            "AMD Radeon RX 7900 XTX|25753026560\n"
        )
        self.assertEqual(cands, [("AMD Radeon RX 7900 XTX", 24), ("AMD Radeon RX 7900 XTX", 24)])

    def test_windows_query_starts_from_present_adapters(self):
        # Stale registry entries for removed cards must not be enumerated on their own
        self.assertIn("Win32_VideoController", speedtest._WIN_GPU_PS)
        self.assertLess(
            speedtest._WIN_GPU_PS.index("Get-CimInstance Win32_VideoController"),
            speedtest._WIN_GPU_PS.index("ForEach-Object"),
        )


class TierRunnerTests(unittest.TestCase):
    def test_long_context_timeout_scales_with_measured_prefill_speed(self):
        base, est = speedtest.estimate_tier_timeout(262144, None)
        self.assertEqual(base, 900.0)
        self.assertIsNone(est)
        # Slow box: 300 tok/s prefill at 128k → 256k needs far more than the fixed 900 s
        t, est = speedtest.estimate_tier_timeout(262144, {"ctx_prompt": 131072, "pp_tps": 300})
        self.assertGreater(est, 1500)
        self.assertGreater(t, est)
        self.assertLessEqual(t, speedtest.MAX_REQUEST_TIMEOUT)
        # Fast box keeps the default
        t, _ = speedtest.estimate_tier_timeout(262144, {"ctx_prompt": 131072, "pp_tps": 20000})
        self.assertEqual(t, 900.0)

    def test_shared_runner_keeps_finished_tiers_and_uses_log_callback(self):
        lines: List[str] = []
        calls: List[Dict[str, Any]] = []

        def fake_bc(endpoint, model, target, gen, runs, **kw):
            calls.append({"target": target, "timeout": kw.get("timeout")})
            if target > 40000:
                raise RuntimeError("out of memory")
            return {"ctx_prompt": target, "pp_tps": 300.0, "tg_tps": 50.0, "gen_tokens": gen}, 5.6

        with mock.patch.object(speedtest, "benchmark_context", side_effect=fake_bc), \
                mock.patch.object(speedtest, "run_single_completion", return_value={"source": "timings"}) as warm:
            res = speedtest.run_benchmark_tiers(
                "http://127.0.0.1:1/v1", "m", [1024, 32768, 65536, 131072], 256, 3,
                warmup=True, log=lambda m, c=None: lines.append(m),
            )
        self.assertEqual(warm.call_count, 1)
        self.assertEqual([r["ctx_prompt"] for r in res], [1024, 32768])
        self.assertEqual(len(calls), 3)  # stops after the 64k failure
        self.assertGreater(calls[2]["timeout"], 600.0)  # 64k timeout scaled from the 32k prefill speed
        self.assertTrue(any("停止更高档位" in ln for ln in lines))

    def test_first_tier_failure_is_raised(self):
        with mock.patch.object(speedtest, "benchmark_context", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                speedtest.run_benchmark_tiers("http://127.0.0.1:1/v1", "m", [1024], 256, 3, warmup=False,
                                              log=lambda m, c=None: None)


class EdgeCaseTests(unittest.TestCase):
    """Pending review notices, saved payloads + --resubmit, busy server detection, multi-model hint."""

    def setUp(self):
        import os
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = mock.patch.dict(os.environ, {"LLM_SPEEDTEST_CLIENT_ID_FILE": os.path.join(self._tmp.name, "client_id")})
        env.start()
        self.addCleanup(env.stop)

    def _server(self, routes):
        """routes: {(method, path): callable(body) -> (status, json_obj)}"""
        calls: List[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _handle(self, method):
                n = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else None
                path = self.path.split("?")[0]
                calls.append(f"{method} {path}")
                fn = routes.get((method, path))
                status, obj = fn(body) if fn else (404, {"error": "not found"})
                raw = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}", calls

    def test_review_notice_explains_flagged_and_dropped(self):
        notes = speedtest.review_notice({
            "results": [
                {"ctx_tier": "1k", "status": "flagged", "flag_reason": "outlier_vs_median:5.95x"},
                {"ctx_tier": "32k", "status": "approved"},
            ],
            "dropped": [{"index": 2, "error": "results[2] 数值字段无效或超出范围"}],
        })
        self.assertEqual(len(notes), 2)
        self.assertIn("[1k]", notes[0])
        self.assertIn("明显高于同配置其他用户", notes[0])
        self.assertIn("人工审核", notes[0])
        self.assertIn("1 档数据未通过", notes[1])
        self.assertEqual(speedtest.review_notice({"results": [{"status": "approved"}]}), [])

    def test_ceremony_shows_pending_review_instead_of_congrats(self):
        payload = {"hardware": {"hw_raw": "RTX 4090"}, "software": {}, "model": {"model_raw": "m"},
                   "results": [{"ctx_prompt": 1024, "tg_tps": 1000}]}
        resp = {"record_id": 9, "results": [{"ctx_tier": "1k", "status": "flagged", "flag_reason": "unknown_hw_id:rtx9090"}]}
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            speedtest.print_upload_ceremony(payload, resp, None)
        out = buf.getvalue()
        self.assertIn("等待人工审核", out)
        self.assertIn("硬件型号还没有收录", out)
        self.assertNotIn("成功上榜", out)

    def test_failed_upload_is_saved_and_resubmit_uploads_it(self):
        state = {"fail": True, "got": []}

        def submit(body):
            if state["fail"]:
                return 503, {"ok": False, "error": "服务暂时不可用"}
            state["got"].append(body)
            return 200, {"ok": True, "record_id": 1, "results": [{"id": 1, "status": "approved", "ctx_tier": "1k"}]}

        base, _ = self._server({("POST", "/submit"): submit})
        payload = {"schema_version": 1, "hardware": {"hw_raw": "x"}, "model": {"model_raw": "m"},
                   "results": [{"ctx_prompt": 1024, "tg_tps": 50}]}
        path = speedtest.save_pending_payload(payload)
        self.assertTrue(path and path.endswith(".json"))

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            code = speedtest.upload_payload(payload, f"{base}/submit", None, saved_path=path)
        self.assertEqual(code, 1)
        self.assertIn("--resubmit", buf.getvalue())
        self.assertEqual(speedtest.find_latest_pending(), path)

        state["fail"] = False
        with mock.patch("sys.stderr", io.StringIO()):
            code = speedtest.cli(["--resubmit", "--submit-url", f"{base}/submit", "--stats-url", ""])
        self.assertEqual(code, 0)
        self.assertEqual(state["got"][0]["results"][0]["tg_tps"], 50)
        self.assertIsNone(speedtest.find_latest_pending())  # marked as uploaded

        with mock.patch("sys.stderr", io.StringIO()) as err:
            self.assertEqual(speedtest.cli(["--resubmit"]), 1)
        self.assertIn("没有找到待上传的成绩", err.getvalue())

    def test_server_busy_reads_llamacpp_slots(self):
        base, _ = self._server({("GET", "/slots"): lambda b: (200, [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}])})
        self.assertTrue(speedtest.server_busy(f"{base}/v1"))
        base2, _ = self._server({("GET", "/slots"): lambda b: (200, [{"id": 0, "state": 0}])})
        self.assertFalse(speedtest.server_busy(f"{base2}/v1"))
        base3, _ = self._server({})
        self.assertIsNone(speedtest.server_busy(f"{base3}/v1"))  # /slots disabled or not llama.cpp

    def test_busy_warning_is_emitted_before_benchmark(self):
        lines: List[str] = []
        with mock.patch.object(speedtest, "server_busy", return_value=True), \
                mock.patch.object(speedtest, "benchmark_context", return_value=({"ctx_prompt": 1024, "pp_tps": 1}, 5.6)):
            speedtest.run_benchmark_tiers("http://127.0.0.1:1/v1", "m", [1024], 64, 1, warmup=False,
                                          log=lambda m, c=None: lines.append(m))
        self.assertTrue(any("正在处理其他请求" in ln for ln in lines))

    def test_non_interactive_run_names_the_model_and_warns_about_others(self):
        def chat(body):
            return 200, {}

        base, _ = self._server({("GET", "/v1/models"): lambda b: (200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})})
        err = io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch("sys.stdout", io.StringIO()), \
                mock.patch.object(speedtest, "run_benchmark_tiers", return_value=[]):
            speedtest.cli(["--endpoint", f"{base}/v1", "--non-interactive", "--dry-run", "--hw-id", "rtx4090"])
        out = err.getvalue()
        self.assertIn("测试模型：", out)
        self.assertIn("model-a", out)
        self.assertIn("--model", out)


class ServerNoticeTests(unittest.TestCase):
    def test_nickname_notice_and_sybil_reason_are_shown(self):
        notes = speedtest.review_notice({
            "results": [{"ctx_tier": "1k", "status": "flagged", "flag_reason": "too_many_client_ids:4"}],
            "notices": ["署名「ITBANANA官方」包含保留字（如官方、管理员、ITBANANA），已改为匿名展示"],
        })
        self.assertTrue(any("多个匿名 ID" in n for n in notes))
        self.assertTrue(any("已改为匿名展示" in n for n in notes))

    def test_notice_only_keeps_congrats_title(self):
        payload = {"hardware": {"hw_raw": "RTX 4090"}, "software": {}, "model": {"model_raw": "m"},
                   "results": [{"ctx_prompt": 1024, "tg_tps": 170}]}
        resp = {"record_id": 3, "results": [{"ctx_tier": "1k", "status": "approved"}],
                "notices": ["署名「官方」包含保留字（如官方、管理员、ITBANANA），已改为匿名展示"]}
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            speedtest.print_upload_ceremony(payload, resp, None)
        out = buf.getvalue()
        self.assertNotIn("等待人工审核", out)
        self.assertIn("已改为匿名展示", out)


class PrefillEstimateTests(unittest.TestCase):
    """Engines without server timings: subtract the fixed per-request overhead from TTFT."""

    def _serve(self, overhead_s=0.15, prefill_tps=5000.0, with_timings=False):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                prompt_tokens = max(1, round(len(body["messages"][0]["content"]) / 5.6))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(overhead_s + prompt_tokens / prefill_tps)  # 固定开销 + 真实读题耗时
                n = body["max_tokens"]
                for _ in range(n):
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"tok "}}]}\n\n')
                    self.wfile.flush()
                    time.sleep(0.001)
                final = {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n}}
                if with_timings:
                    final["timings"] = {"prompt_n": prompt_tokens, "predicted_n": n,
                                        "prompt_per_second": prefill_tps, "predicted_per_second": 900.0}
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1"

    def test_overhead_correction_brings_1k_prefill_close_to_truth(self):
        ep = self._serve(overhead_s=0.15, prefill_tps=5000.0)
        overhead = speedtest.measure_ttft_overhead(ep, "m")
        self.assertIsNotNone(overhead)
        self.assertAlmostEqual(overhead, 0.15, delta=0.06)
        prompt = speedtest.build_prompt(1024, "t")
        raw = speedtest.run_single_completion(ep, "m", prompt, 8)
        fixed = speedtest.run_single_completion(ep, "m", prompt, 8, ttft_overhead_s=overhead)
        self.assertLess(raw["pp_tps"], 5000 * 0.7)  # 不扣开销：明显低估
        self.assertGreater(fixed["pp_tps"], 5000 * 0.8)  # 扣除后接近真实 5000
        self.assertLess(fixed["pp_tps"], 5000 * 1.25)
        self.assertEqual(raw["ttft_ms"] // 50, fixed["ttft_ms"] // 50)  # 首字时间本身照实上报，不做修改

    def test_unreliable_overhead_falls_back_to_plain_ttft(self):
        ep = self._serve(overhead_s=0.0, prefill_tps=200000.0)
        r = speedtest.run_single_completion(ep, "m", speedtest.build_prompt(1024, "t"), 8, ttft_overhead_s=10.0)
        self.assertGreater(r["pp_tps"], 0)
        self.assertLess(r["pp_tps"], 200000 * 1.5)

    def test_no_probe_when_server_reports_timings(self):
        ep = self._serve(with_timings=True)
        self.assertIsNone(speedtest.measure_ttft_overhead(ep, "m"))
        r = speedtest.run_single_completion(ep, "m", "hello", 8, ttft_overhead_s=0.15)
        self.assertEqual(r["source"], "timings")
        self.assertEqual(r["pp_tps"], 5000.0)

    def test_runner_measures_overhead_only_for_sse_engines(self):
        lines: List[str] = []
        ep = self._serve(overhead_s=0.1, prefill_tps=8000.0)
        res = speedtest.run_benchmark_tiers(ep, "m", [1024], 16, 1, warmup=True, log=lambda m, c=None: lines.append(m))
        self.assertTrue(any("固定开销约" in ln for ln in lines))
        self.assertGreater(res[0]["pp_tps"], 8000 * 0.75)

        lines2: List[str] = []
        ep2 = self._serve(with_timings=True)
        speedtest.run_benchmark_tiers(ep2, "m", [1024], 16, 1, warmup=True, log=lambda m, c=None: lines2.append(m))
        self.assertFalse(any("固定开销约" in ln for ln in lines2))


class CeremonyLayoutTests(unittest.TestCase):
    """The card must never be wider than the terminal (borders would wrap and the box falls apart)."""

    def _render(self, cols):
        payload = {
            "hardware": {"hw_raw": "AMD Radeon AI PRO R9700", "cpu_raw": "AMD Ryzen 7 5700X", "kind": "gpu", "ram_gb": 64, "ram_type": "DDR4"},
            "software": {}, "model": {"model_raw": "qwen3.8-flash-next-iq3_s", "quant": "GSQ-RCO IQ3_S"},
            "results": [{"ctx_prompt": c, "tg_tps": 100.0, "pp_tps": 3000.0, "ttft_ms": 9000.0} for c in (1024, 32768, 65536, 131072, 262144)],
        }
        resp = {"record_id": 65, "total_samples": 60, "total_contributors": 5, "ranking": {
            "same_hw_rank": 1, "same_hw_total": 4, "same_hw_median": 116.9, "vs_median_pct": 4,
            "same_model_rank": 1, "same_model_total": 5, "gap_to_first": 0, "submitted_tg": 120.3,
            "personal_best": 121, "is_personal_best": False, "vs_personal_best": -0.7, "personal_best_record_id": 51,
            "ladder_url": "https://itbanana.com/llm-speed?highlight=65"}}
        stats = {"group": {"tg_min": 114.6, "tg_max": 121, "tg_median": 116.9, "contributors": 4}}
        buf = io.StringIO()
        with mock.patch.object(speedtest, "_terminal_columns", return_value=cols), mock.patch("sys.stderr", buf):
            speedtest.print_upload_ceremony(payload, resp, stats)
        return [ln for ln in buf.getvalue().splitlines() if ln.strip()]

    def test_card_fits_narrow_terminal_and_keeps_url_whole(self):
        for cols in (80, 100, 120):
            lines = self._render(cols)
            widths = {speedtest._display_width(ln) for ln in lines if ln.startswith(("╔", "╠", "╟", "╚"))}
            self.assertEqual(len(widths), 1, cols)
            self.assertLessEqual(max(widths), cols, cols)
            for ln in lines:
                self.assertLessEqual(speedtest._display_width(ln), cols, (cols, ln))
            self.assertTrue(any(ln.strip() == "https://itbanana.com/llm-speed?highlight=65" or ln.strip().endswith("https://itbanana.com/llm-speed?highlight=65") for ln in lines))
            joined = "".join(ln.strip() for ln in lines)
            self.assertIn("未刷新纪录", joined)
            self.assertIn("查看本次成绩 #65", joined)

    def test_wrap_display_handles_cjk_and_ansi(self):
        line = "  ★ " + speedtest._c("同配置同模型速度排名", "1;33") + "：第 1 / 4 名（个人最佳较同配置中位 116.9 tok/s +4%） · 本次较个人历史最佳低 0.7 tok/s（最佳 121 tok/s），未刷新纪录"
        parts = speedtest._wrap_display(line, 60)
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLessEqual(speedtest._display_width(p), 60)
        self.assertEqual(speedtest.strip_ansi("".join(p.strip() for p in parts)).replace(" ", ""),
                         speedtest.strip_ansi(line).replace(" ", ""))


class TitleSemanticsTests(unittest.TestCase):
    """首位开荒 / 刷新个人纪录 / 卫冕 / 新王登基 must reflect what actually happened."""

    payload = {"hardware": {"hw_raw": "GPU"}, "software": {}, "model": {"model_raw": "m"},
               "results": [{"ctx_prompt": 1024, "tg_tps": 1}]}

    def _out(self, ranking, **resp_extra):
        base = {"same_model_rank": 1, "same_model_total": 5, "gap_to_first": 0, "is_personal_best": True,
                "vs_personal_best": 0, "personal_best_record_id": 9}
        resp = {"record_id": 9, "total_samples": 69, "total_contributors": 2, "ranking": {**base, **ranking}, **resp_extra}
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf), mock.patch.object(speedtest, "_terminal_columns", return_value=200):
            speedtest.print_upload_ceremony(self.payload, resp, None)
        return buf.getvalue()

    def test_5070ti_case_sole_returning_user_is_new_pb_not_pioneer(self):
        out = self._out({"same_hw_rank": 1, "same_hw_total": 1, "submitted_tg": 71.8, "personal_best": 71.8,
                         "prev_personal_best": 71.6, "vs_prev_best": 0.2, "was_top": True})
        self.assertNotIn("首位开荒", out)
        self.assertNotIn("首个基准", out)
        self.assertIn("刷新个人纪录", out)
        self.assertIn("+0.2 tok/s", out)
        self.assertIn("该配置目前仅你一人实测", out)

    def test_r9700_case_already_top_is_defend_not_new_king(self):
        out = self._out({"same_hw_rank": 1, "same_hw_total": 2, "same_hw_median": 121.2, "vs_median_pct": 2,
                         "submitted_tg": 123.6, "personal_best": 123.6,
                         "prev_personal_best": 121.0, "vs_prev_best": 2.6, "was_top": True})
        self.assertNotIn("新王登基", out)
        self.assertIn("卫冕榜首", out)
        self.assertIn("+2.6 tok/s", out)

    def test_overtaking_someone_else_is_new_king(self):
        out = self._out({"same_hw_rank": 1, "same_hw_total": 3, "submitted_tg": 125, "personal_best": 125,
                         "prev_personal_best": 115, "vs_prev_best": 10, "was_top": False})
        self.assertIn("新王登基", out)

    def test_first_ever_on_config_is_pioneer(self):
        out = self._out({"same_hw_rank": 1, "same_hw_total": 1, "submitted_tg": 60, "personal_best": 60,
                         "prev_personal_best": None, "vs_prev_best": None, "was_top": False})
        self.assertIn("首位开荒", out)
        self.assertIn("同配置首个基准", out)

    def test_runner_up_improving_own_record(self):
        out = self._out({"same_hw_rank": 2, "same_hw_total": 3, "gap_to_first": 4, "submitted_tg": 110,
                         "personal_best": 110, "prev_personal_best": 105, "vs_prev_best": 5, "was_top": False})
        self.assertIn("刷新个人纪录", out)
        self.assertIn("距同配置第一仅差 4 tok/s", out)

    def test_contributor_line_says_which_contributor_you_are(self):
        out = self._out({"same_hw_rank": 1, "same_hw_total": 1, "prev_personal_best": None},
                        total_runs=23, contributor_no=1)
        self.assertIn("全站已汇聚 23 次真实测速（2 位贡献者），你是第 1 位加入的贡献者", out)
        self.assertNotIn("感谢第 2 位贡献者", out)
        old = self._out({"same_hw_rank": 1, "same_hw_total": 1})
        self.assertIn("条测速记录，来自 2 位贡献者", old)


class RemeasureTests(unittest.TestCase):
    """A tier slower than a longer-context tier is re-measured once (256k never), best result kept."""

    def _run(self, tg_by_ctx, rerun_tg):
        calls: List[int] = []
        lines: List[str] = []

        def fake_bc(endpoint, model, target, gen, runs, **kw):
            calls.append(target)
            first = calls.count(target) == 1
            tg = tg_by_ctx[target] if first else rerun_tg.get(target, tg_by_ctx[target])
            return {"ctx_prompt": target, "tg_tps": tg, "pp_tps": 3000.0, "gen_tokens": 256, "runs": 1 if target >= 65536 else 3}, 5.6

        ctxs = list(tg_by_ctx)
        with mock.patch.object(speedtest, "benchmark_context", side_effect=fake_bc), \
                mock.patch.object(speedtest, "server_busy", return_value=False), \
                mock.patch.object(speedtest, "measure_ttft_overhead", return_value=None):
            res = speedtest.run_benchmark_tiers("http://127.0.0.1:1/v1", "m", ctxs, 256, 3, ctx_limit=300000,
                                                warmup=False, log=lambda m, c=None: lines.append(m))
        return res, calls, lines

    def test_r9700_case_64k_dip_is_remeasured_and_better_result_kept(self):
        # #70 的真实数据：64k 87.5 比 256k 91.3 低 4.2%
        tg = {1024: 123.6, 32768: 107.05, 65536: 87.5, 131072: 89.3, 262144: 91.3}
        res, calls, lines = self._run(tg, {65536: 100.2})
        self.assertEqual(calls.count(65536), 2)
        self.assertEqual(calls.count(262144), 1)  # 256k 不重测
        self.assertEqual(calls.count(131072), 1)  # 128k 比 256k 只低 2.2%，在正常波动范围内
        r64 = next(r for r in res if r["ctx_prompt"] == 65536)
        self.assertEqual(r64["tg_tps"], 100.2)
        self.assertEqual(r64["runs"], 2)
        self.assertTrue(any("正在重测该档一次" in ln for ln in lines))
        self.assertTrue(any("采用重测成绩 100.2" in ln for ln in lines))

    def test_remeasure_keeps_original_when_not_better(self):
        tg = {1024: 120, 65536: 85, 131072: 92}
        res, calls, lines = self._run(tg, {65536: 84})
        self.assertEqual(calls.count(65536), 2)
        self.assertEqual(next(r for r in res if r["ctx_prompt"] == 65536)["tg_tps"], 85)
        self.assertTrue(any("保留原成绩 85" in ln for ln in lines))

    def test_normal_small_wiggle_and_monotonic_curves_are_not_remeasured(self):
        # 5070 Ti 的真实数据：32k 63.65 比 64k 65.3 低 2.6%，属于正常波动
        tg = {1024: 71.8, 32768: 63.65, 65536: 65.3, 131072: 64.3}
        _, calls, lines = self._run(tg, {})
        self.assertEqual(len(calls), 4)
        self.assertFalse(any("重测" in ln for ln in lines))

    def test_no_max_wait_wording_in_logs(self):
        lines: List[str] = []
        with mock.patch.object(speedtest, "benchmark_context",
                               side_effect=lambda e, m, t, g, r, **kw: ({"ctx_prompt": t, "tg_tps": 50.0, "pp_tps": 100.0}, 5.6)), \
                mock.patch.object(speedtest, "server_busy", return_value=False), \
                mock.patch.object(speedtest, "measure_ttft_overhead", return_value=None):
            speedtest.run_benchmark_tiers("http://127.0.0.1:1/v1", "m", [1024, 262144], 256, 1, ctx_limit=300000,
                                          warmup=False, log=lambda m, c=None: lines.append(m))
        self.assertTrue(any("预计每轮约" in ln for ln in lines))
        self.assertFalse(any("最长等待" in ln for ln in lines))
