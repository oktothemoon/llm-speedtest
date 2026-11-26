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


class EndToEndMockTests(unittest.TestCase):
    def setUp(self):
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
                    prompt_len = max(1, round(len(body["messages"][0]["content"]) / 4))
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
        self.assertEqual(len(p["results"]), 2)
        err_text = err_buf.getvalue()
        self.assertIn("新王登基", err_text)
        self.assertIn("实测编号 #1", err_text)
        self.assertIn("https://itbanana.com/llm-speed?highlight=1", err_text)

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
            self.assertEqual(st["payload"]["software"]["kv_cache"], "int8")
            self.assertEqual(st["payload"]["software"]["ctx_limit"], 270000)
            self.assertEqual(len(st["payload"]["results"]), 3)
            self.assertEqual(st["payload"]["results"][0]["tg_tps"], 170.5)

            # 6. Upload through UI backend
            up = speedtest.http_json(f"{ui_base}/api/upload", payload={"payload": st["payload"]}, headers=hdrs)
            self.assertTrue(up["ok"])
            self.assertEqual(len(self.uploaded), 1)
            self.assertEqual(self.uploaded[0]["hardware"]["detected_by"], "local")
        finally:
            ui_srv.shutdown()
            ui_srv.server_close()


if __name__ == "__main__":
    unittest.main()
