#!/usr/bin/env python3
"""
Zero-dependency local web UI for llm-speedtest (Python 3.9+ stdlib only).
Binds strictly to 127.0.0.1 with a per-session token on all POST endpoints.
"""

from __future__ import annotations

import http.server
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional


def _catalog_items(st: Any) -> Dict[str, Any]:
    gpus = []
    unified = []
    for hid, label in st.HW_ID_LABELS.items():
        if hid in st.UNIFIED_IDS:
            unified.append({"id": hid, "label": label})
        else:
            gpus.append({"id": hid, "label": label, "vram_gb": st.HW_DEFAULT_VRAM.get(hid)})
    return {"gpus": gpus, "unified": unified}


def _render_html(token: str, default_submit_url: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>llm-speedtest · 本地大模型测速</title>
<style>
  :root {{
    --bg: #f6f7f9;
    --card: #ffffff;
    --line: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --accent: #0f766e;
    --accent-hover: #115e59;
    --accent-soft: #ccfbf1;
    --warn-bg: #fffbeb;
    --warn-border: #fde68a;
    --warn-text: #92400e;
    --ok-bg: #ecfdf5;
    --ok-text: #065f46;
    --err-bg: #fef2f2;
    --err-text: #991b1b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 780px; margin: 28px auto 60px; padding: 0 16px; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; gap: 12px; flex-wrap: wrap; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 13px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .card-title {{ font-size: 15px; font-weight: 600; margin: 0 0 12px; display: flex; align-items: center; justify-content: space-between; }}
  .seg {{ display: inline-flex; background: #eef1f4; padding: 3px; border-radius: 10px; gap: 3px; }}
  .seg button {{
    border: 0; background: transparent; padding: 6px 14px; border-radius: 8px;
    font-size: 13px; font-weight: 500; color: var(--muted); cursor: pointer;
  }}
  .seg button.active {{ background: #fff; color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  .row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }}
  .row-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .row-1 {{ grid-template-columns: 1fr; }}
  @media (max-width: 640px) {{ .row, .row-3 {{ grid-template-columns: 1fr; }} }}
  label {{ display: block; font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 5px; }}
  input, select {{
    width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 9px;
    font-size: 14px; background: #fff; color: var(--text); outline: none;
  }}
  input:focus, select:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(15,118,110,0.12); }}
  .inline {{ display: flex; gap: 8px; align-items: center; }}
  .inline > input, .inline > select {{ flex: 1; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
  .chip {{
    border: 1px solid var(--line); background: #fff; color: var(--text);
    padding: 5px 10px; border-radius: 999px; font-size: 12px; cursor: pointer;
  }}
  .chip:hover {{ border-color: var(--accent); }}
  .chip.active {{ background: var(--accent-soft); border-color: var(--accent); color: var(--accent-hover); font-weight: 600; }}
  .btn {{
    border: 0; border-radius: 10px; padding: 9px 16px; font-size: 14px; font-weight: 600;
    cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-primary:hover {{ background: var(--accent-hover); }}
  .btn-primary:disabled {{ opacity: 0.55; cursor: not-allowed; }}
  .btn-ghost {{ background: #eef1f4; color: var(--text); }}
  .btn-ghost:hover {{ background: #e2e6eb; }}
  .btn-lg {{ width: 100%; padding: 12px 18px; font-size: 15px; }}
  .banner {{
    border-radius: 10px; padding: 10px 12px; font-size: 13px; margin-bottom: 12px;
  }}
  .banner-warn {{ background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-text); }}
  .banner-ok {{ background: var(--ok-bg); color: var(--ok-text); }}
  .banner-err {{ background: var(--err-bg); color: var(--err-text); }}
  .hidden {{ display: none !important; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; font-size: 13px; color: var(--muted); user-select: none; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }}
  .metric {{ background: #f9fafb; border: 1px solid var(--line); border-radius: 10px; padding: 12px; text-align: center; }}
  .metric .v {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
  .metric .k {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  pre {{
    background: #111827; color: #e5e7eb; padding: 12px 14px; border-radius: 10px;
    font-size: 12px; overflow-x: auto; max-height: 280px; margin: 10px 0 12px;
  }}
  .log {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); white-space: pre-wrap; margin-top: 10px; }}
  .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .actions .btn-primary {{ flex: 1; }}
  .privacy {{ font-size: 12px; color: var(--muted); margin-top: 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>llm-speedtest</h1>
      <div class="sub">本地大模型标准化测速 · 零个人信息采集 · 数据汇总至 itbanana.com/llm-hardware</div>
    </div>
    <div class="seg" role="tablist">
      <button id="tab-local" class="active" type="button">测本机</button>
      <button id="tab-lan" type="button">测局域网另一台电脑</button>
    </div>
  </header>

  <!-- Step 1: Server & Model -->
  <section class="card">
    <div class="card-title">
      <span>1. 推理服务与模型</span>
      <span id="conn-badge" class="sub">检测中...</span>
    </div>

    <div id="lan-bar" class="hidden">
      <div class="banner banner-warn">
        已关闭本机硬件自动探测，避免把当前电脑配置错记成局域网推理机配置。同 IP 下次会自动记住你选过的硬件。
      </div>
      <div class="row row-1">
        <div>
          <label for="endpoint-input">局域网推理机 IP 或接口地址（直接填 IP 也行，自动探端口和路径）</label>
          <div class="inline">
            <input id="endpoint-input" placeholder="例如 192.168.31.88 或 192.168.31.88:11434 或完整 URL" />
            <button id="btn-probe" class="btn btn-primary" type="button">连接</button>
          </div>
        </div>
      </div>
      <details>
        <summary>进阶：这台机器支持免密 SSH？点此一键自动识别远端硬件（SSH 地址绝不上传）</summary>
        <div class="inline" style="margin-top:8px;">
          <input id="ssh-input" placeholder="例如 user@192.168.31.88" />
          <button id="btn-ssh" class="btn btn-ghost" type="button">SSH 识别硬件</button>
        </div>
      </details>
    </div>

    <div id="local-bar">
      <div class="row row-1">
        <div>
          <label for="local-endpoint">本机接口地址（已自动扫描，也可改端口后点重新连接）</label>
          <div class="inline">
            <input id="local-endpoint" placeholder="http://127.0.0.1:8080/v1" />
            <button id="btn-local-probe" class="btn btn-ghost" type="button">重新检测</button>
          </div>
        </div>
      </div>
    </div>

    <div id="probe-err" class="banner banner-err hidden"></div>

    <div class="row" style="margin-top:10px;">
      <div>
        <label for="model-select">已加载模型</label>
        <select id="model-select"></select>
      </div>
      <div>
        <label for="quant-input">量化版本（自动识别，可按需修改）</label>
        <input id="quant-input" placeholder="例如 GSQ-RCO IQ3_S / Q4_K_M / FP8" />
      </div>
    </div>
  </section>

  <!-- Step 2: Hardware & Software -->
  <section class="card">
    <div class="card-title">
      <span>2. 硬件与推理引擎</span>
      <span id="hw-hint" class="sub"></span>
    </div>

    <div id="quick-hw-wrap" class="hidden" style="margin-bottom:12px;">
      <label>热门配置一键点选（选完自动填显卡和显存）</label>
      <div class="chips" id="quick-hw">
        <button type="button" class="chip" data-hw="rtx4090-48">RTX 4090 48G</button>
        <button type="button" class="chip" data-hw="rtx5090">RTX 5090</button>
        <button type="button" class="chip" data-hw="rtx5080">RTX 5080</button>
        <button type="button" class="chip" data-hw="rx7900xtx">RX 7900 XTX</button>
        <button type="button" class="chip" data-hw="r9700">R9700 32G</button>
        <button type="button" class="chip" data-hw="rtx3090">RTX 3090</button>
        <button type="button" class="chip" data-hw="rtx4080s-32">RTX 4080S 32G</button>
        <button type="button" class="chip" data-hw="m5ultra">Mac M5 Ultra</button>
        <button type="button" class="chip" data-hw="m5max40">Mac M5 Max</button>
      </div>
    </div>

    <div class="row row-3">
      <div>
        <label for="hw-select">显卡 / 芯片</label>
        <select id="hw-select"></select>
      </div>
      <div id="hw-custom-wrap" class="hidden">
        <label for="hw-raw-input">自定义型号名称</label>
        <input id="hw-raw-input" placeholder="例如 RTX 4090D 48GB" />
      </div>
      <div id="gpu-count-wrap">
        <label>显卡张数</label>
        <div class="chips" id="gpu-count-chips">
          <button type="button" class="chip active" data-val="1">1 张</button>
          <button type="button" class="chip" data-val="2">2 张</button>
          <button type="button" class="chip" data-val="4">4 张</button>
          <button type="button" class="chip" data-val="8">8 张</button>
        </div>
      </div>
      <div>
        <label for="ram-gb">系统 / 统一内存 (GB)</label>
        <input id="ram-gb" type="number" min="8" max="8192" placeholder="如 64 / 128" />
        <div class="chips" id="ram-quick">
          <button type="button" class="chip" data-ram="32">32G</button>
          <button type="button" class="chip" data-ram="64">64G</button>
          <button type="button" class="chip" data-ram="128">128G</button>
          <button type="button" class="chip" data-ram="256">256G</button>
        </div>
      </div>
    </div>

    <div class="row row-3">
      <div id="ram-type-wrap">
        <label for="ram-preset">内存规格（选填）</label>
        <select id="ram-preset">
          <option value="">不确定 / 跳过</option>
          <option value="DDR4|2">DDR4 双通道</option>
          <option value="DDR5|2">DDR5 双通道</option>
          <option value="DDR4|4">X99/X299 四通道 DDR4</option>
          <option value="DDR3|4">X99 四通道 DDR3</option>
          <option value="DDR4|8">八通道 DDR4 工作站</option>
          <option value="DDR5|8">八通道 DDR5 工作站</option>
          <option value="DDR5|12">十二通道 DDR5 服务器</option>
          <option value="Unified|">统一内存 (Mac / AI Max)</option>
        </select>
      </div>
      <div>
        <label for="engine-select">推理引擎（选填）</label>
        <select id="engine-select">
          <option value="">不确定 / 跳过</option>
          <option value="strata">Strata</option>
          <option value="llamacpp">llama.cpp / Ollama / LM Studio</option>
          <option value="vllm">vLLM</option>
          <option value="sglang">SGLang</option>
          <option value="mlx">MLX / oMLX / mlx-serve</option>
          <option value="maya">Project Maya</option>
        </select>
      </div>
      <div>
        <label>MTP / 推测解码</label>
        <div class="chips" id="mtp-chips">
          <button type="button" class="chip active" data-val="">不确定</button>
          <button type="button" class="chip" data-val="true">开启</button>
          <button type="button" class="chip" data-val="false">关闭</button>
        </div>
      </div>
    </div>

    <details>
      <summary>高级设置（上下文长度、轮数、操作系统、API Key）</summary>
      <div class="row row-3" style="margin-top:10px;">
        <div>
          <label for="ctx-input">测试上下文 (tokens)</label>
          <input id="ctx-input" value="1024" placeholder="1024 或 1024,32768" />
        </div>
        <div>
          <label for="runs-input">每档轮数（取中位数）</label>
          <input id="runs-input" type="number" min="1" max="10" value="3" />
        </div>
        <div>
          <label for="os-select">目标机系统</label>
          <select id="os-select">
            <option value="">自动 / 跳过</option>
            <option value="linux">Linux</option>
            <option value="windows">Windows</option>
            <option value="macos">macOS</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label for="vram-input">单卡显存 GB（仅魔改/自定义卡需填）</label>
          <input id="vram-input" type="number" min="1" max="2048" placeholder="自动" />
        </div>
        <div>
          <label for="apikey-input">本地服务 API Key（如有，绝不上传）</label>
          <input id="apikey-input" type="password" placeholder="选填" />
        </div>
      </div>
    </details>
  </section>

  <!-- Step 3: Run & Upload -->
  <section class="card">
    <button id="btn-run" class="btn btn-primary btn-lg" type="button">开始测速</button>
    <div id="run-log" class="log hidden"></div>

    <div id="result-box" class="hidden" style="margin-top:14px;">
      <div class="score-grid" id="score-grid"></div>
      <div class="card-title" style="margin-top:10px;">
        <span>即将上传的完整明文 JSON（请核对）</span>
        <button id="btn-copy" class="btn btn-ghost" style="padding:5px 10px;font-size:12px;" type="button">复制 JSON</button>
      </div>
      <pre id="json-preview"></pre>
      <div id="upload-msg" class="banner hidden"></div>
      <div class="actions">
        <button id="btn-upload" class="btn btn-primary" type="button">确认上传至 itbanana.com</button>
      </div>
      <div class="privacy">
        隐私保证：仅上传上方明文 JSON 中的硬件规格与速度指标，绝不采集用户名、主机名、路径、序列号或原始 IP。上传目标：{default_submit_url}
      </div>
    </div>
  </section>
</div>

<script>
const TOKEN = {json.dumps(token)};
const $ = (id) => document.getElementById(id);

let mode = 'local'; // 'local' | 'lan'
let localHw = null;
let catalog = {{ gpus: [], unified: [] }};
let gpuCount = 1;
let mtpVal = '';
let lastPayload = null;
let pollTimer = null;

async function api(path, body) {{
  const opts = body === undefined
    ? {{ method: 'GET' }}
    : {{ method: 'POST', headers: {{ 'Content-Type': 'application/json', 'X-Speedtest-Token': TOKEN }}, body: JSON.stringify(body) }};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}}

function fillHwDropdown() {{
  const sel = $('hw-select');
  sel.innerHTML = '';
  const addOpt = (parent, val, text) => {{
    const o = document.createElement('option'); o.value = val; o.textContent = text; parent.appendChild(o);
  }};
  addOpt(sel, '', '请选择显卡 / 芯片...');
  const g1 = document.createElement('optgroup'); g1.label = '独立显卡 (GPU)';
  for (const g of catalog.gpus) addOpt(g1, g.id, g.label);
  sel.appendChild(g1);
  const g2 = document.createElement('optgroup'); g2.label = '统一内存 (Mac / AI Max / DGX Spark)';
  for (const u of catalog.unified) addOpt(g2, u.id, u.label);
  sel.appendChild(g2);
  addOpt(sel, '__custom__', '其他未收录型号（手动输入）');
}}

function isUnifiedId(id) {{
  return catalog.unified.some((u) => u.id === id);
}}

function applyHwToForm(hw) {{
  if (!hw) return;
  if (hw.hw_id && [...$('hw-select').options].some((o) => o.value === hw.hw_id)) {{
    $('hw-select').value = hw.hw_id;
    $('hw-custom-wrap').classList.add('hidden');
  }} else if (hw.hw_raw) {{
    $('hw-select').value = '__custom__';
    $('hw-raw-input').value = hw.hw_raw;
    $('hw-custom-wrap').classList.remove('hidden');
  }} else {{
    $('hw-select').value = '';
    $('hw-custom-wrap').classList.add('hidden');
  }}
  setGpuCount(hw.gpu_count || 1);
  $('vram-input').value = hw.vram_gb || '';
  $('ram-gb').value = hw.ram_gb || '';
  $('os-select').value = hw.os || '';
  if (hw.ram_type === 'Unified') $('ram-preset').value = 'Unified|';
  else if (hw.ram_type && hw.ram_channels) $('ram-preset').value = hw.ram_type + '|' + hw.ram_channels;
  else $('ram-preset').value = '';
  updateUnifiedVisibility();
}}

function updateUnifiedVisibility() {{
  const id = $('hw-select').value;
  const uni = isUnifiedId(id);
  $('gpu-count-wrap').classList.toggle('hidden', uni);
  if (uni) {{
    $('ram-preset').value = 'Unified|';
    if (id.startsWith('m') && !$('os-select').value) $('os-select').value = 'macos';
  }}
  $('hw-custom-wrap').classList.toggle('hidden', id !== '__custom__');
}}

function setGpuCount(n) {{
  gpuCount = Number(n) || 1;
  document.querySelectorAll('#gpu-count-chips .chip').forEach((b) => b.classList.toggle('active', Number(b.dataset.val) === gpuCount));
}}

function setMtp(v) {{
  mtpVal = v;
  document.querySelectorAll('#mtp-chips .chip').forEach((b) => b.classList.toggle('active', b.dataset.val === v));
}}

function setModels(models, guessedEngine, guessedQuant) {{
  const sel = $('model-select');
  sel.innerHTML = '';
  for (const m of models) {{
    const o = document.createElement('option'); o.value = m; o.textContent = m; sel.appendChild(o);
  }}
  if (guessedEngine && !$('engine-select').value) $('engine-select').value = guessedEngine;
  if (guessedQuant) $('quant-input').value = guessedQuant;
}}

function saveLanMemory(ep) {{
  try {{
    const key = 'llm-speedtest:lan:' + ep.trim();
    const rec = {{
      hw_id: $('hw-select').value,
      hw_raw: $('hw-raw-input').value,
      gpu_count: gpuCount,
      vram_gb: $('vram-input').value,
      ram_gb: $('ram-gb').value,
      ram_preset: $('ram-preset').value,
      engine: $('engine-select').value,
      mtp: mtpVal,
      os: $('os-select').value,
    }};
    localStorage.setItem(key, JSON.stringify(rec));
  }} catch {{}}
}}

function loadLanMemory(ep) {{
  try {{
    const raw = localStorage.getItem('llm-speedtest:lan:' + ep.trim());
    if (!raw) return false;
    const rec = JSON.parse(raw);
    if (rec.hw_id) $('hw-select').value = rec.hw_id;
    if (rec.hw_raw) $('hw-raw-input').value = rec.hw_raw;
    setGpuCount(rec.gpu_count || 1);
    $('vram-input').value = rec.vram_gb || '';
    $('ram-gb').value = rec.ram_gb || '';
    $('ram-preset').value = rec.ram_preset || '';
    $('engine-select').value = rec.engine || '';
    setMtp(rec.mtp || '');
    $('os-select').value = rec.os || '';
    updateUnifiedVisibility();
    return true;
  }} catch {{ return false; }}
}}

function switchMode(next) {{
  mode = next;
  $('tab-local').classList.toggle('active', mode === 'local');
  $('tab-lan').classList.toggle('active', mode === 'lan');
  $('local-bar').classList.toggle('hidden', mode !== 'local');
  $('lan-bar').classList.toggle('hidden', mode !== 'lan');
  $('quick-hw-wrap').classList.toggle('hidden', mode !== 'lan');
  $('probe-err').classList.add('hidden');
  if (mode === 'local') {{
    applyHwToForm(localHw);
    $('hw-hint').textContent = '已自动识别本机硬件';
  }} else {{
    $('hw-select').value = '';
    $('hw-raw-input').value = '';
    $('vram-input').value = '';
    $('ram-gb').value = '';
    $('ram-preset').value = '';
    $('os-select').value = '';
    setGpuCount(1);
    updateUnifiedVisibility();
    $('hw-hint').textContent = '请选择局域网推理机配置';
    const ep = $('endpoint-input').value.trim();
    if (ep && loadLanMemory(ep)) $('hw-hint').textContent = '已自动载入上次记住的该主机配置';
  }}
}}

async function probe(rawEndpoint) {{
  $('probe-err').classList.add('hidden');
  $('conn-badge').textContent = '连接中...';
  try {{
    const res = await api('/api/probe', {{ endpoint: rawEndpoint, api_key: $('apikey-input').value.trim() || null }});
    if (mode === 'lan') {{
      $('endpoint-input').value = res.endpoint;
      if (loadLanMemory(res.endpoint) || loadLanMemory(rawEndpoint)) {{
        $('hw-hint').textContent = '已自动载入上次记住的该主机配置';
      }}
    }} else {{
      $('local-endpoint').value = res.endpoint;
    }}
    setModels(res.models, res.guessed_engine, res.guessed_quant);
    $('conn-badge').textContent = '已连接 · ' + res.models.length + ' 个模型';
  }} catch (e) {{
    $('conn-badge').textContent = '未连接';
    $('probe-err').textContent = e.message;
    $('probe-err').classList.remove('hidden');
  }}
}}

async function init() {{
  try {{
    const data = await api('/api/init');
    catalog = data.catalog;
    localHw = data.local_hw;
    fillHwDropdown();
    applyHwToForm(localHw);
    $('hw-hint').textContent = '已自动识别本机硬件';
    if (data.local_endpoint) {{
      $('local-endpoint').value = data.local_endpoint;
      setModels(data.local_models, data.guessed_engine, data.guessed_quant);
      $('conn-badge').textContent = '已自动连上本机服务';
    }} else {{
      $('conn-badge').textContent = '未检测到本机服务（可填端口或切到局域网）';
    }}
  }} catch (e) {{
    $('conn-badge').textContent = '初始化失败：' + e.message;
  }}
}}

$('tab-local').onclick = () => switchMode('local');
$('tab-lan').onclick = () => switchMode('lan');
$('btn-probe').onclick = () => probe($('endpoint-input').value.trim());
$('btn-local-probe').onclick = () => probe($('local-endpoint').value.trim());
$('hw-select').onchange = () => {{
  const id = $('hw-select').value;
  const g = catalog.gpus.find((x) => x.id === id);
  if (g && g.vram_gb) $('vram-input').value = g.vram_gb;
  else if (isUnifiedId(id)) $('vram-input').value = '';
  updateUnifiedVisibility();
}};

document.querySelectorAll('#quick-hw .chip').forEach((btn) => {{
  btn.onclick = () => {{
    $('hw-select').value = btn.dataset.hw;
    $('hw-select').onchange();
    document.querySelectorAll('#quick-hw .chip').forEach((b) => b.classList.toggle('active', b === btn));
  }};
}});
document.querySelectorAll('#gpu-count-chips .chip').forEach((b) => b.onclick = () => setGpuCount(b.dataset.val));
document.querySelectorAll('#mtp-chips .chip').forEach((b) => b.onclick = () => setMtp(b.dataset.val));
document.querySelectorAll('#ram-quick .chip').forEach((b) => b.onclick = () => {{ $('ram-gb').value = b.dataset.ram; }});

$('btn-ssh').onclick = async () => {{
  const target = $('ssh-input').value.trim();
  if (!target) return;
  $('hw-hint').textContent = '正在通过 SSH 读取远端硬件...';
  try {{
    const res = await api('/api/ssh-detect', {{ ssh: target }});
    applyHwToForm(res.hw);
    $('hw-hint').textContent = '已通过 SSH 自动识别远端硬件';
  }} catch (e) {{
    $('hw-hint').textContent = 'SSH 识别失败：' + e.message;
  }}
}};

$('model-select').onchange = async () => {{
  const m = $('model-select').value;
  try {{
    const res = await api('/api/infer-model', {{ model: m }});
    if (res.quant) $('quant-input').value = res.quant;
    if (res.engine && !$('engine-select').value) $('engine-select').value = res.engine;
  }} catch {{}}
}};

$('btn-run').onclick = async () => {{
  const ep = (mode === 'lan' ? $('endpoint-input').value : $('local-endpoint').value).trim();
  const model = $('model-select').value;
  const hwId = $('hw-select').value;
  const hwRaw = hwId === '__custom__' ? $('hw-raw-input').value.trim() : ($('hw-select').selectedOptions[0]?.textContent || '');
  if (!ep || !model) {{
    alert('请先连接推理服务并选择模型');
    return;
  }}
  if (mode === 'lan' && !hwId) {{
    alert('请先点选局域网推理机的显卡或芯片');
    return;
  }}
  if (mode === 'lan') saveLanMemory(ep);

  const [ramType, ramCh] = ($('ram-preset').value || '|').split('|');
  const body = {{
    endpoint: ep,
    api_key: $('apikey-input').value.trim() || null,
    model,
    quant: $('quant-input').value.trim() || null,
    engine: $('engine-select').value || null,
    mtp: mtpVal === '' ? null : mtpVal === 'true',
    ctx: $('ctx-input').value.trim() || '1024',
    runs: Number($('runs-input').value) || 3,
    remote: mode === 'lan',
    hardware: {{
      kind: isUnifiedId(hwId) ? 'unified' : 'gpu',
      hw_id: hwId && hwId !== '__custom__' ? hwId : null,
      hw_raw: hwRaw || (localHw && localHw.hw_raw) || 'Unknown',
      gpu_count: isUnifiedId(hwId) ? 1 : gpuCount,
      vram_gb: $('vram-input').value ? Number($('vram-input').value) : null,
      ram_gb: $('ram-gb').value ? Number($('ram-gb').value) : null,
      ram_type: ramType || null,
      ram_channels: ramCh ? Number(ramCh) : null,
      os: $('os-select').value || (mode === 'local' && localHw ? localHw.os : null),
    }},
  }};

  $('btn-run').disabled = true;
  $('btn-run').textContent = '测速中...';
  $('result-box').classList.add('hidden');
  $('upload-msg').classList.add('hidden');
  $('run-log').textContent = '正在预热并执行测速，请稍候（约 15–40 秒）...';
  $('run-log').classList.remove('hidden');

  try {{
    await api('/api/start', body);
    clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 700);
  }} catch (e) {{
    $('run-log').textContent = '启动失败：' + e.message;
    $('btn-run').disabled = false;
    $('btn-run').textContent = '开始测速';
  }}
}};

async function pollStatus() {{
  try {{
    const st = await api('/api/status');
    if (st.logs && st.logs.length) $('run-log').textContent = st.logs.join('\\n');
    if (st.state === 'done') {{
      clearInterval(pollTimer);
      $('btn-run').disabled = false;
      $('btn-run').textContent = '重新测速';
      lastPayload = st.payload;
      renderResult(st.payload);
    }} else if (st.state === 'error') {{
      clearInterval(pollTimer);
      $('btn-run').disabled = false;
      $('btn-run').textContent = '重新测速';
      $('run-log').textContent = (st.logs || []).join('\\n') + '\\n错误：' + st.error;
    }}
  }} catch {{}}
}}

function renderResult(p) {{
  const r = p.results[0];
  $('score-grid').innerHTML = `
    <div class="metric"><div class="v">${{r.tg_tps}}</div><div class="k">生成速度 (tok/s)</div></div>
    <div class="metric"><div class="v">${{r.pp_tps || '-'}}</div><div class="k">读题速度 (tok/s)</div></div>
    <div class="metric"><div class="v">${{r.ttft_ms || '-'}} ms</div><div class="k">首字延迟 (${{r.source}})</div></div>
  `;
  $('json-preview').textContent = JSON.stringify(p, null, 2);
  $('result-box').classList.remove('hidden');
  $('btn-upload').disabled = false;
  $('btn-upload').textContent = '确认上传至 itbanana.com';
}}

$('btn-copy').onclick = () => {{
  if (!lastPayload) return;
  navigator.clipboard?.writeText(JSON.stringify(lastPayload, null, 2));
  $('btn-copy').textContent = '已复制';
  setTimeout(() => {{ $('btn-copy').textContent = '复制 JSON'; }}, 1500);
}};

$('btn-upload').onclick = async () => {{
  if (!lastPayload) return;
  $('btn-upload').disabled = true;
  $('btn-upload').textContent = '上传中...';
  $('upload-msg').classList.add('hidden');
  try {{
    const res = await api('/api/upload', {{ payload: lastPayload }});
    let msg = '上传成功！感谢贡献实测数据。';
    if (res.stats && res.stats.group && res.stats.group.contributors) {{
      const g = res.stats.group;
      msg += ` 当前同配置已有 ${{g.contributors}} 位贡献者，实测区间 ${{g.tg_min}}–${{g.tg_max}} tok/s（中位 ${{g.tg_median}}）。`;
    }}
    $('upload-msg').className = 'banner banner-ok';
    $('upload-msg').textContent = msg;
    $('upload-msg').classList.remove('hidden');
    $('btn-upload').textContent = '已上传';
  }} catch (e) {{
    $('upload-msg').className = 'banner banner-err';
    $('upload-msg').textContent = '上传失败：' + e.message;
    $('upload-msg').classList.remove('hidden');
    $('btn-upload').disabled = false;
    $('btn-upload').textContent = '重试上传';
  }}
}};

init();
</script>
</body>
</html>
"""


def create_ui_server(
    st_module: Any,
    submit_url: str,
    stats_url: str,
    port: int = 0,
    gen_tokens_default: int = 256,
    no_warmup: bool = False,
) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Create a 127.0.0.1-only local web UI server and return (server, session_token)."""
    token = secrets.token_hex(16)
    html_bytes = _render_html(token, submit_url).encode("utf-8")
    lock = threading.Lock()
    job: Dict[str, Any] = {"state": "idle", "logs": [], "payload": None, "error": None}

    def run_job(cfg: Dict[str, Any]) -> None:
        def log(msg: str) -> None:
            with lock:
                job["logs"].append(msg)

        try:
            endpoint = cfg["endpoint"].rstrip("/")
            model = cfg["model"]
            api_key = cfg.get("api_key")
            runs = max(1, min(10, int(cfg.get("runs") or 3)))
            gen_tokens = max(16, min(4096, int(cfg.get("gen_tokens") or gen_tokens_default)))
            ctx_list = [int(x.strip()) for x in str(cfg.get("ctx") or "1024").split(",") if x.strip()]
            if not ctx_list:
                ctx_list = [1024]

            if not no_warmup:
                log("正在发送预热请求...")
                st_module.run_single_completion(endpoint, model, "Say OK.", 24, api_key=api_key)

            results: List[Dict[str, Any]] = []
            for c in ctx_list:
                samples = []
                for i in range(runs):
                    nonce = f"{c}-{i}-{int(time.time() * 1000) % 100000}"
                    prompt = st_module.build_prompt(c, nonce)
                    log(f"[ctx≈{c}] 正在跑第 {i + 1}/{runs} 轮...")
                    r = st_module.run_single_completion(endpoint, model, prompt, gen_tokens, api_key=api_key)
                    samples.append(r)
                    log(f"[ctx≈{c}] 第 {i + 1}/{runs} 轮完成：生成 {r['tg_tps']} tok/s，读题 {r['pp_tps']} tok/s（{r['source']}）")
                import statistics
                results.append({
                    "ctx_prompt": int(round(statistics.median(s["prompt_tokens"] for s in samples))),
                    "gen_tokens": int(round(statistics.median(s["gen_tokens"] for s in samples))),
                    "tg_tps": round(statistics.median(s["tg_tps"] for s in samples), 2),
                    "pp_tps": round(statistics.median(s["pp_tps"] for s in samples), 2),
                    "sse_tg_tps": round(statistics.median(s["sse_tg_tps"] for s in samples), 2),
                    "ttft_ms": round(statistics.median(s["ttft_ms"] for s in samples), 1),
                    "runs": runs,
                    "source": "timings" if all(s["source"] == "timings" for s in samples) else "sse",
                })

            hw = cfg["hardware"]
            payload = {
                "schema_version": st_module.SCHEMA_VERSION,
                "client_version": st_module.CLIENT_VERSION,
                "hardware": {
                    "kind": hw.get("kind") or "gpu",
                    "hw_id": hw.get("hw_id"),
                    "hw_raw": hw.get("hw_raw") or "Unknown Hardware",
                    "gpu_count": int(hw.get("gpu_count") or 1),
                    "vram_gb": hw.get("vram_gb"),
                    "ram_gb": hw.get("ram_gb"),
                    "ram_type": hw.get("ram_type"),
                    "ram_channels": hw.get("ram_channels"),
                    "os": hw.get("os"),
                },
                "software": {
                    "engine": cfg.get("engine"),
                    "engine_version": None,
                    "backend": None,
                    "mtp": cfg.get("mtp"),
                    "ctx_limit": None,
                },
                "model": {
                    "model_id": st_module.map_model_id(model),
                    "model_raw": model,
                    "quant": cfg.get("quant") or st_module.infer_quant(model),
                    "fit": None,
                },
                "results": results,
            }
            log("测速完成！")
            with lock:
                job["state"] = "done"
                job["payload"] = payload
        except Exception as e:
            with lock:
                job["state"] = "error"
                job["error"] = str(e)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_json(self, status: int, data: Any) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            return json.loads(raw)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif path == "/api/init":
                local_hw = st_module.detect_hardware()
                ep = None
                models: List[str] = []
                try:
                    ep, models = st_module.discover_endpoint()
                except Exception:
                    pass
                first_m = models[0] if models else ""
                self._send_json(200, {
                    "catalog": _catalog_items(st_module),
                    "local_hw": local_hw,
                    "local_endpoint": ep,
                    "local_models": models,
                    "guessed_engine": st_module.infer_engine_auto(ep or "", first_m) if ep else None,
                    "guessed_quant": st_module.infer_quant(first_m) if first_m else None,
                })
            elif path == "/api/status":
                with lock:
                    self._send_json(200, dict(job))
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.headers.get("X-Speedtest-Token") != token:
                self._send_json(403, {"error": "Invalid session token"})
                return
            path = urllib.parse.urlparse(self.path).path
            try:
                body = self._read_json()
            except Exception:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            if path == "/api/probe":
                raw_ep = (body.get("endpoint") or "").strip()
                try:
                    ep, models = st_module.resolve_endpoint(raw_ep, api_key=body.get("api_key"))
                    first_m = models[0] if models else ""
                    self._send_json(200, {
                        "endpoint": ep,
                        "models": models,
                        "remote": not st_module.is_local_endpoint(ep),
                        "guessed_engine": st_module.infer_engine_auto(ep, first_m),
                        "guessed_quant": st_module.infer_quant(first_m),
                    })
                except Exception as e:
                    self._send_json(400, {"error": str(e)})
            elif path == "/api/ssh-detect":
                target = (body.get("ssh") or "").strip()
                hw = st_module.detect_hardware_via_ssh(target) if target else None
                if not hw:
                    self._send_json(400, {"error": "未能免密 SSH 识别，请直接下拉框点选配置"})
                else:
                    self._send_json(200, {"hw": hw})
            elif path == "/api/infer-model":
                m = body.get("model") or ""
                self._send_json(200, {
                    "model_id": st_module.map_model_id(m),
                    "quant": st_module.infer_quant(m),
                    "engine": st_module.infer_engine_auto("", m),
                })
            elif path == "/api/start":
                with lock:
                    if job["state"] == "running":
                        self._send_json(409, {"error": "已有测速正在进行中"})
                        return
                    job["state"] = "running"
                    job["logs"] = ["开始测速..."]
                    job["payload"] = None
                    job["error"] = None
                t = threading.Thread(target=run_job, args=(body,), daemon=True)
                t.start()
                self._send_json(200, {"ok": True})
            elif path == "/api/upload":
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "缺少 payload"})
                    return
                try:
                    resp = st_module.http_json(submit_url, payload=payload, timeout=15.0)
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", errors="ignore")
                    self._send_json(502, {"error": f"HTTP {e.code}: {detail}"})
                    return
                except Exception as e:
                    self._send_json(502, {"error": str(e)})
                    return

                stats = None
                hw_id = (payload.get("hardware") or {}).get("hw_id")
                model_id = (payload.get("model") or {}).get("model_id")
                gpu_count = (payload.get("hardware") or {}).get("gpu_count") or 1
                if hw_id and model_id and stats_url:
                    try:
                        q = urllib.parse.urlencode({"hw_id": hw_id, "model_id": model_id, "gpu_count": gpu_count})
                        stats = st_module.http_json(f"{stats_url}?{q}", timeout=5.0)
                    except Exception:
                        pass
                self._send_json(200, {"ok": True, "response": resp, "stats": stats})
            else:
                self._send_json(404, {"error": "Not found"})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, token


def serve_ui(st_module: Any, submit_url: str, stats_url: str, port: int = 0, open_browser: bool = True) -> int:
    server, _ = create_ui_server(st_module, submit_url=submit_url, stats_url=stats_url, port=port)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}"
    print(f"llm-speedtest 本地测速界面已启动：{url}", file=sys.stderr)
    print("按 Ctrl+C 退出。", file=sys.stderr)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
    finally:
        server.server_close()
    return 0
