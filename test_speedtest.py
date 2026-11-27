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
