#!/usr/bin/env python3
"""Unit & end-to-end mock server tests for llm-speedtest."""

import http.server
import json
import threading
import time
import unittest
from typing import Any, Dict, List

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

    def test_remote_endpoint_guard(self):
        self.assertTrue(speedtest.is_local_endpoint("http://127.0.0.1:8080/v1"))
        self.assertTrue(speedtest.is_local_endpoint("http://localhost:11434/v1"))
        self.assertTrue(speedtest.is_local_endpoint("http://[::1]:8080/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://192.168.1.50:8080/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://10.0.0.8:8000/v1"))
        self.assertFalse(speedtest.is_local_endpoint("http://gpu-box.local:8080/v1"))

        # Remote endpoint without explicit hardware must be rejected (never leaks local hardware)
        rc = speedtest.main(["--endpoint", "http://192.168.1.50:8080/v1", "--dry-run"])
        self.assertEqual(rc, 2)

        # Remote endpoint WITH explicit hardware must not leak local OS / RAM
        args = speedtest.parse_args([
            "--endpoint", "http://192.168.1.50:8080/v1",
            "--hw-id", "rtx4090-48",
            "--ram-gb", "64",
            "--ram-type", "DDR4",
            "--os", "linux",
        ])
        hw = speedtest.resolve_hardware(args, remote=True)
        self.assertEqual(hw["kind"], "gpu")
        self.assertEqual(hw["hw_id"], "rtx4090-48")
        self.assertEqual(hw["hw_raw"], "RTX 4090 48GB")
        self.assertEqual(hw["ram_gb"], 64)
        self.assertEqual(hw["os"], "linux")


class EndToEndMockTests(unittest.TestCase):
    def setUp(self):
        self.uploaded: List[Dict[str, Any]] = []
        uploaded_ref = self.uploaded

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send_json(200, {"data": [{"id": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"}]})
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
                        "usage": {"prompt_tokens": 1024, "completion_tokens": 20},
                        "timings": {
                            "prompt_n": 1024,
                            "predicted_n": 20,
                            "prompt_per_second": 3800.0,
                            "predicted_per_second": 170.5,
                        },
                    }
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                elif self.path == "/api/llm-speedtest/submit":
                    uploaded_ref.append(body)
                    self._send_json(200, {"ok": True, "batch_id": "test-batch", "accepted": len(body["results"])})
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

    def test_end_to_end_benchmark_and_upload(self):
        base = f"http://127.0.0.1:{self.port}"
        rc = speedtest.main([
            "--endpoint", f"{base}/v1",
            "--hw-id", "rtx4090-48",
            "--hw-raw", "NVIDIA GeForce RTX 4090",
            "--gpu-count", "1",
            "--vram-gb", "48",
            "--ram-gb", "64",
            "--ram-type", "DDR4",
            "--ram-channels", "2",
            "--engine", "strata",
            "--mtp", "true",
            "--fit", "hybrid",
            "--ctx", "1024",
            "--gen-tokens", "20",
            "--runs", "2",
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
        self.assertEqual(p["model"]["model_id"], "qwen38-flash-next")
        self.assertEqual(p["model"]["quant"], "GSQ-RCO IQ3_S")
        self.assertEqual(p["results"][0]["tg_tps"], 170.5)
        self.assertEqual(p["results"][0]["source"], "timings")


if __name__ == "__main__":
    unittest.main()
