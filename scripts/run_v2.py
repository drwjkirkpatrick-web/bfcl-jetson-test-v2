#!/usr/bin/env python3
"""
BFCL v2 Jetson Test — Model runner.

For each working model from v1 (>= 50% overall):
  1. Kill any running llama-server
  2. Launch llama-server with --jinja (+ template override where needed)
  3. Wait for readiness
  4. Run bfcl_v2_harness.py (50 tool tests + 2 coding tests)
  5. Kill server, save results

Memory notes (Jetson Orin Nano 8GB, GDM off):
  - Coding tests need 16384 max_tokens -> context must exceed that, so
    coding-capable models get -c 24576. Small models get -c 8192 for
    tool tests only... but simpler: everyone gets -c 24576 with q8_0 KV
    (24K * 2 * 1B/8 = ~6GB? no - q8_0 KV for 3B model ~= 24576 * 2 *
    1.5KB = too much). Context budget:
      3B-class model, q8_0 KV: ~0.75 MB/token-pair -> 24K tokens ~= 18GB. NO.
  Actually: KV cache size = 2 (K+V) * n_layers * n_kv_heads * head_dim *
  2 bytes (f16) per token. For Qwen2.5-3B: 36 layers * 2 KV heads * 128
  dim * 2 * 2 = ~36.8KB/token f16 -> 24K tokens ~= 0.9GB f16, ~0.45GB q8_0.
  That fits. Large 8B models at q8_0: ~2.4x more -> 24K ~= 1.1GB q8_0. Fits.
  We launch once per model with -c 24576 -ctk q8_0 -ctv q8_0 and run
  everything against that single server.

Usage:
  python3 run_v2.py                     # all models
  python3 run_v2.py --models granite4-3b,xlam-2-1b
  python3 run_v2.py --skip-existing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = PROJECT_DIR / "templates"
ARCHIVE_DIR = Path.home() / "projects" / "bfcl-jetson-test-v2-results"
ARCHIVE_DIR.mkdir(exist_ok=True)
LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
HARNESS = PROJECT_DIR / "scripts" / "bfcl_v2_harness.py"
PORT = 8090
HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# Model registry — the 17 v1 models that scored >= 50%
# template: None = built-in; otherwise a file in templates/
# ctx: context size (coding needs > 16384 for max_tokens headroom)
# ngl: GPU layers (99 = all; hybrid SSM models can't offload attention)
# ---------------------------------------------------------------------------

MODELS = [
    # v1 champions (>90%)
    {"name": "granite4-3b",   "path": "/home/walker/models/new-zoo/granite4-3b.gguf",
     "template": None, "ctx": 24576, "ngl": 15, "note": "hybrid SSM: limited GPU offload"},
    {"name": "granite4.1-3b", "path": "/home/walker/models/new-zoo/granite4.1-3b.gguf",
     "template": None, "ctx": 24576, "ngl": 15},
    {"name": "granite4.2-3b", "path": "/home/walker/models/new-zoo/granite-4.2-3b-Q4_K_M.gguf",
     "template": None, "ctx": 24576, "ngl": 15},
    {"name": "granite3.2-2b", "path": "/home/walker/models/new-zoo/granite3.2-2b.gguf",
     "template": None, "ctx": 24576, "ngl": 15},
    {"name": "nanbeige4-3b",  "path": "/home/walker/projects/bfcl-jetson-test/models/Nanbeige_Nanbeige4-3B-Thinking-2511-Q8_0.gguf",
     "template": None, "ctx": 24576, "ngl": 99, "note": "thinking model"},
    {"name": "ternary-bonsai-q4",  "path": "/home/walker/projects/bfcl-jetson-test/models/Ternary-Bonsai-8B-Q4_0-lossless.gguf",
     "template": None, "ctx": 16384, "ngl": 99, "kv": "q4_0", "note": "4.3GB: q4_0 KV to fit", "coding_max_tokens": 12288},
    {"name": "ternary-bonsai-tq2", "path": "/home/walker/projects/bfcl-jetson-test/models/Ternary-Bonsai-8B-TQ2_0.gguf",
     "template": None, "ctx": 24576, "ngl": 99},
    {"name": "xlam-2-1b",     "path": "/home/walker/projects/bfcl-jetson-test/models/xLAM-2-1B-fc-r-Q8_0.gguf",
     "template": "xlam2-qwen25-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "xlam-2-3b",     "path": "/home/walker/projects/bfcl-jetson-test/models/xLAM-2-3B-fc-r-Q8_0.gguf",
     "template": "xlam2-qwen25-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "xlam-2-8b",     "path": "/home/walker/projects/bfcl-jetson-test/models/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf",
     "template": None, "ctx": 16384, "ngl": 99, "kv": "q4_0", "note": "4.9GB: q4_0 KV to fit", "coding_max_tokens": 12288},
    {"name": "arch-agent-1.5b", "path": "/home/walker/projects/bfcl-jetson-test/models/Arch-Agent-1.5B-q8_0.gguf",
     "template": "xlam2-qwen25-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "qwen2.5-3b",    "path": "/home/walker/models/new-zoo/qwen2.5-3b.gguf",
     "template": None, "ctx": 24576, "ngl": 99},
    {"name": "hermes3-3b-q5", "path": "/home/walker/models/hermes3-3b-q5_k_m.gguf",
     "template": "hermes3-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "hermes3-3b-q4", "path": "/home/walker/models/hermes3-3b-q4_k_m.gguf",
     "template": "hermes3-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "llama3.2-3b",   "path": "/home/walker/models/new-zoo/llama3.2-3b-bench.gguf",
     "template": None, "ctx": 24576, "ngl": 15, "note": "hybrid: limited offload"},
    {"name": "hammer2.1-3b",  "path": "/home/walker/projects/bfcl-jetson-test/models/Hammer2.1-3b.Q8_0.gguf",
     "template": "hermes3-tool-use.jinja", "ctx": 24576, "ngl": 99},
    {"name": "qwen3.5-9b",    "path": "/home/walker/models/Qwen_Qwen3.5-9B-Q3_K_S.gguf",
     "template": None, "ctx": 16384, "ngl": 99, "kv": "q4_0", "note": "4.1GB DeltaNet hybrid: q4_0 KV to fit", "coding_max_tokens": 12288},
]


def kill_llama_server() -> None:
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    check = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
    if check.stdout.strip():
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(3)


def wait_ready(port: int, timeout: int = 180) -> bool:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def launch_server(model: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    cmd = [
        LLAMA_SERVER,
        "-m", model["path"],
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", str(model["ngl"]),
        "-c", str(model["ctx"]),
        "-ctk", model.get("kv", "q8_0"),
        "-ctv", model.get("kv", "q8_0"),
        "-b", "512", "-ub", "512",
        "-fa", "on",
        "--jinja",
        "-np", "1",
        "-t", "6",
        "--temp", "0.0",
        "--top-k", "1",
        "--repeat-penalty", "1.0",
        "--fit", "off",
    ]
    if model.get("template"):
        cmd += ["--chat-template-file", str(TEMPLATES_DIR / model["template"])]
    log = RESULTS_DIR / f"server_{model['name']}.log"
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    return proc


def run_harness(model_name: str, timeout_sec: int = 14400) -> dict:
    """Run the full v2 harness for one model (up to 4h)."""
    output = RESULTS_DIR / f"{model_name}.json"
    cmd = [
        sys.executable, str(HARNESS),
        "--host", HOST, "--port", str(PORT),
        "--model", model_name,
        "--output", str(output),
        "--timeout", "180",
        "--coding-timeout", "6000",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        print(r.stdout[-3000:])
        if r.stderr:
            print(r.stderr[-500:], file=sys.stderr)
        return {"ok": r.returncode == 0, "output": str(output)}
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout_sec}s")
        return {"ok": False, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tool-tests-only", action="store_true", help="Skip coding tests")
    args = parser.parse_args()

    models = MODELS
    if args.models:
        names = set(args.models.split(","))
        models = [m for m in MODELS if m["name"] in names]

    print(f"BFCL v2 runner — {len(models)} models")
    for m in models:
        name = m["name"]
        result_file = RESULTS_DIR / f"{name}.json"
        if args.skip_existing and result_file.exists():
            print(f"SKIP {name}: results exist")
            continue
        if not os.path.exists(m["path"]):
            print(f"SKIP {name}: file not found")
            continue
        print("\n" + "=" * 64)
        print(f"[{name}] ctx={m['ctx']} ngl={m['ngl']} "
              f"template={m.get('template') or 'built-in'}")
        print("=" * 64)
        kill_llama_server()
        proc = launch_server(m)
        if not wait_ready(PORT):
            print(f"  ERROR: server failed to start; see results/server_{name}.log")
            proc.kill()
            continue
        print("  Server READY")
        cmd = [sys.executable, str(HARNESS), "--host", HOST, "--port", str(PORT),
               "--model", name, "--output", str(result_file),
               "--timeout", "180"]
        if args.tool_tests_only:
            cmd.append("--tool-tests-only")
        else:
            cmd += ["--coding-timeout", "6000",
                    "--coding-max-tokens", str(m.get("coding_max_tokens", 16384))]
        harness_args = []
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            print(r.stdout[-2500:])
            if r.returncode != 0:
                print(f"  HARNESS CRASHED (exit {r.returncode}):")
                print("  " + r.stderr[-1200:].replace("\n", "\n  "))
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT — model took > 2h")
        proc.kill()
        kill_llama_server()
        # Archive a copy of this model's results outside the repo for analysis
        if result_file.exists():
            import shutil
            shutil.copy2(result_file, ARCHIVE_DIR / result_file.name)
            print(f"  Archived -> {ARCHIVE_DIR / result_file.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
