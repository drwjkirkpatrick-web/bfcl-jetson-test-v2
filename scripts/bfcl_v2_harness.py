#!/usr/bin/env python3
"""
BFCL v2 Jetson Test — Harness.

Runs against a llama.cpp llama-server exposing an OpenAI-compatible API
(--jinja). Evaluates four categories:

  1. simple             (30) — single call, tricky argument extraction
  2. moderate_parallel  (12) — 2-3 independent calls, order-independent match
  3. moderate_chained    (8) — multi-step with mock tool execution fed back
  4. coding             (2)  — Python tic-tac-toe + HTML profile page,
                               auto-graded by execution + static checks

Evaluation follows BFCL-style AST rules: strict on required params,
lenient on extras, string normalization, int/float coercion.

Usage:
  python3 bfcl_v2_harness.py --port 8090 --model mymodel
  python3 bfcl_v2_harness.py --port 8090 --model mymodel --coding-only
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import os
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests_v2 import SIMPLE_TESTS, MODERATE_PARALLEL, MODERATE_CHAINED, TOOLS

# ---------------------------------------------------------------------------
# Evaluation (BFCL-style, ported from v1)
# ---------------------------------------------------------------------------

def normalize_string(s: str) -> str:
    """Normalize a string for comparison per BFCL rules."""
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r'[,./\-_*^()]', '', s)
    return s


def values_match(predicted, expected) -> bool:
    """BFCL AST-style value matching with type coercion."""
    # Bool: strict
    if isinstance(expected, bool):
        if isinstance(predicted, bool):
            return predicted is expected
        if isinstance(predicted, str):
            return normalize_string(predicted) in ("true", "false") and \
                   (normalize_string(predicted) == "true") == expected
        return False
    # Numbers: int/float coercion
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return float(predicted) == float(expected)
    # Strings: normalized
    if isinstance(expected, str) and isinstance(predicted, str):
        return normalize_string(predicted) == normalize_string(expected)
    # Lists: recursive, order-insensitive for ingredient-style lists
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(expected) != len(predicted):
            return False
        if all(values_match(p, e) for p, e in zip(predicted, expected)):
            return True
        # order-insensitive fallback
        remaining = list(predicted)
        for e in expected:
            for i, p in enumerate(remaining):
                if values_match(p, e):
                    remaining.pop(i)
                    break
            else:
                return False
        return True
    # Dicts: key presence + value match
    if isinstance(expected, dict) and isinstance(predicted, dict):
        for k, v in expected.items():
            if k not in predicted:
                return False
            if not values_match(predicted[k], v):
                return False
        return True
    # Fallback
    return predicted == expected


def evaluate_call(predicted: dict, expected: dict) -> bool:
    """Single tool-call match: name + required args (lenient on extras)."""
    if predicted.get("name") != expected["name"]:
        return False
    pred_args = predicted.get("args", {})
    exp_args = expected.get("args", {})
    for param_name, exp_val in exp_args.items():
        if param_name not in pred_args:
            return False
        if not values_match(pred_args[param_name], exp_val):
            return False
    return True


def evaluate_response(predicted_calls: list, expected_calls: list) -> bool:
    """Order-independent matching for parallel calls."""
    if len(expected_calls) == 0:
        return len(predicted_calls) == 0
    if len(predicted_calls) != len(expected_calls):
        return False
    matched = [False] * len(expected_calls)
    for pred in predicted_calls:
        for i, exp in enumerate(expected_calls):
            if matched[i]:
                continue
            if evaluate_call(pred, exp):
                matched[i] = True
                break
    return all(matched)


# ---------------------------------------------------------------------------
# LLM communication
# ---------------------------------------------------------------------------

def send_request(host, port, messages, tools=None, temperature=0.0,
                 max_tokens=512, timeout=120, model="local-model"):
    """POST /v1/chat/completions with optional tools. One retry on transient connection errors."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True
    import requests as _rq
    last_err = None
    for attempt in (1, 2):  # retry once on connection-level errors only, never on read timeout
        try:
            r = _rq.post(url, json=payload, timeout=timeout)
            return r.json()
        except _rq.exceptions.ConnectionError as e:
            last_err = e
            if attempt == 1:
                time.sleep(3)
                continue
        except Exception as e:
            return {"error": f"request failed: {e}"}
    return {"error": f"connection failed after retry: {last_err}"}


def server_healthy(host, port, timeout=5) -> bool:
    """Quick /health probe used to abort gracefully if the server dies mid-run."""
    import requests as _rq
    try:
        r = _rq.get(f"http://{host}:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def extract_tool_calls(response: dict) -> list:
    """Extract tool calls from an OpenAI-format response.

    Two paths, mirroring BFCL's FC and Prompt modes:
      1. Structured tool_calls field (native function calling)
      2. Text fallback: some models emit tool calls as JSON in content
         (their GGUF template never triggers the autoparser grammar).
         Parse a JSON array/object of {name, arguments|parameters|args}.
    """
    calls = []
    if "error" in response and not response.get("choices"):
        return calls
    choices = response.get("choices", [])
    if not choices:
        return calls
    message = choices[0].get("message", {})

    # Path 1: structured tool_calls
    for tc in message.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append({"name": fn.get("name", ""), "args": args})
    if calls:
        return calls

    # Path 2: text fallback (prompt-mode emission)
    content = message.get("content") or ""
    if content and '"name"' in content:
        stripped = re.sub(r"```(?:json)?", "", content)
        frags = []
        # arrays first (greedy outer brackets), then balanced single objects
        for m in re.finditer(r"\[[\s\S]*\]", stripped):
            frags.append(m.group())
        for start in [m.start() for m in re.finditer(r"\{", stripped)]:
            depth, i = 0, start
            while i < len(stripped):
                if stripped[i] == "{": depth += 1
                elif stripped[i] == "}":
                    depth -= 1
                    if depth == 0:
                        frags.append(stripped[start:i+1])
                        break
                i += 1
        for frag in frags:
            if '"name"' not in frag:
                continue
            try:
                parsed = json.loads(frag)
            except json.JSONDecodeError:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            if not items or not all(isinstance(c, dict) and "name" in c for c in items):
                continue
            for c in items:
                args = c.get("arguments", c.get("parameters", c.get("args", {})))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                candidate = {"name": c["name"], "args": args or {}}
                # dedupe exact duplicates (same name+args) produced when the
                # model emits the same call in both bare and fenced JSON
                if candidate not in calls:
                    calls.append(candidate)
        # only keep a coherent batch: if nothing parsed cleanly, calls stays []
    return calls


def tool_response_message(content: dict) -> dict:
    """Wrap a mock tool result as a role=tool message."""
    return {"role": "tool", "tool_call_id": f"mock_{abs(hash(json.dumps(content))) % 10**6}",
            "content": json.dumps(content)}


# ---------------------------------------------------------------------------
# Incremental save — a partial run is still useful data
# ---------------------------------------------------------------------------

_INCREMENTAL_PATH = None

def _incremental_save(host, port, model, results):
    if _INCREMENTAL_PATH is None:
        return
    Path(_INCREMENTAL_PATH).write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Simple / parallel test runner
# ---------------------------------------------------------------------------

def run_flat_tests(host, port, model, tests, category, results, timeout=120):
    """Run single/parallel call tests. Each test is independently fail-safe."""
    for tc in tests:
        try:
            response = send_request(
                host, port,
                messages=[{"role": "user", "content": tc["question"]}],
                tools=tc["tools"], model=model, timeout=timeout,
            )
            predicted = extract_tool_calls(response)
            ok = evaluate_response(predicted, tc["expected"])
            finish = "?"
            if response.get("choices"):
                finish = response["choices"][0].get("finish_reason", "?")
            err = response.get("error")
        except Exception as e:
            predicted, ok, finish, err = [], False, "exception", str(e)
        results["categories"][category]["total"] += 1
        if ok:
            results["categories"][category]["correct"] += 1
        results["details"].append({
            "id": tc["id"], "category": category, "pass": ok,
            "predicted": predicted, "expected": tc["expected"],
            "finish_reason": finish, **({"error": err} if err else {}),
        })
        print(f"  [{'PASS' if ok else 'FAIL'}] {tc['id']:<12} (finish={finish})", flush=True)
        # Write results incrementally so a crash never loses completed tests
        _incremental_save(host, port, model, results)


# ---------------------------------------------------------------------------
# Chained test runner (multi-step with mock execution)
# ---------------------------------------------------------------------------

def run_chained_tests(host, port, model, tests, results, timeout=120):
    """Run multi-step chained tests: execute mocks, feed results back. Fail-safe per test."""
    for tc in tests:
        messages = [{"role": "user", "content": tc["question"]}]
        step_outcomes = []
        all_ok = True
        error = None
        for step_idx, step in enumerate(tc["steps"]):
            try:
                response = send_request(
                    host, port, messages, tools=tc["tools"], model=model,
                    timeout=timeout,
                )
                predicted = extract_tool_calls(response)
                ok = evaluate_response(predicted, step["expected"])
            except Exception as e:
                predicted, ok, error = [], False, str(e)
            step_outcomes.append({"step": step_idx + 1, "pass": ok,
                                  "predicted": predicted, "expected": step["expected"]})
            if not ok:
                all_ok = False
                break
            # Append the assistant message (with tool_calls) and mock results
            message = response.get("choices", [{}])[0].get("message", {})
            messages.append({
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": message.get("tool_calls"),
            })
            for call in step["expected"]:
                if call["name"] in step.get("mocks", {}):
                    messages.append(tool_response_message(step["mocks"][call["name"]]))
                else:
                    messages.append(tool_response_message({"status": "done"}))
        results["categories"]["moderate_chained"]["total"] += 1
        if all_ok:
            results["categories"]["moderate_chained"]["correct"] += 1
        results["details"].append({
            "id": tc["id"], "category": "moderate_chained", "pass": all_ok,
            "steps": step_outcomes, **({"error": error} if error else {}),
        })
        print(f"  [{'PASS' if all_ok else 'FAIL'}] {tc['id']:<12} ({len(step_outcomes)}/{len(tc['steps'])} steps)", flush=True)
        _incremental_save(host, port, model, results)


# ---------------------------------------------------------------------------
# Coding tests — auto-graded
# ---------------------------------------------------------------------------

def extract_code_fence(text: str) -> tuple:
    """Extract the first code fence from model output. Returns (lang, code)."""
    m = re.search(r"```(\w*)\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return "", text


def grade_python_game(code: str) -> dict:
    """Grade the tic-tac-toe game by execution + static checks.

    Static checks: board display, input validation, win/tie detection.
    Execution: script runs cleanly to exit with piped 'n' input.
    """
    checks = {
        "board_print": bool(re.search(r"print\s*\(", code)),
        "input_validation": bool(re.search(r"(while|if)\s*\(?\s*(not\s+)?\w*(valid|occupied|isdigit|digit|range|available|empty)", code, re.IGNORECASE) or "not in" in code or "else:" in code),
        "win_detection": bool(re.search(r"(win|draw|tie|full|three|diagonal)", code, re.IGNORECASE)),
        "runs_clean": False,
    }

    # Execution test: feed 'n' to exit immediately
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            input="n\n", capture_output=True, text=True, timeout=60,
        )
        checks["runs_clean"] = p.returncode == 0
        checks["_stdout_excerpt"] = p.stdout[:300]
        checks["_stderr_excerpt"] = p.stderr[-300:]
    except subprocess.TimeoutExpired:
        checks["_stderr_excerpt"] = "timeout: game loop never exited (did it ask 'Play again?'?)"
    except Exception as e:
        checks["_stderr_excerpt"] = str(e)[:300]

    score = sum(1 for k in ("board_print", "input_validation", "win_detection", "runs_clean")
                if checks.get(k))
    return {"score": score, "max": 4, "checks": checks}


def grade_html_profile(html: str) -> dict:
    """Grade the HTML profile page via static structural checks (no browser).

    Checks (7): doctype, complete structure, h1 name, form (3 inputs + submit),
    dark theme + @media, script hookup, validation logic.
    """
    h = html.lower()
    checks = {
        "doctype": "<!doctype html" in h,
        "structure": ("<html" in h and "<head" in h and "<body" in h
                      and "</html>" in h),
        "h1_name": "<h1" in h and "alex" in h,
        "form": ("<form" in h and h.count("<input") >= 3 and "submit" in h),
        "style_dark_media": ("<style" in h and "@media" in h and "background" in h
                             and ("#1" in h or "#0" in h or "#2" in h or "dark" in h)),
        "script_hookup": ("<script" in h and ("addeventlistener" in h or "onclick" in h)),
        "validation_logic": (("required" in h or "valid" in h or "empty" in h)
                             and ("@" in h or "email" in h)),
    }
    score = sum(1 for v in checks.values() if v)
    return {"score": score, "max": 7, "checks": checks}


def run_coding_tests(host, port, model, results, prompts_dir, timeout=6000, max_tokens=16384):
    """Run the two coding tests from prompts/*.txt. Each test is fail-safe."""
    results["categories"]["coding"] = {"total": 0, "correct": 0}

    for test_id, prompt_file, grader, pass_at in (
        ("coding_python_game", "python_game_prompt.txt", grade_python_game, 3),
        ("coding_html_profile", "html_profile_prompt.txt", grade_html_profile, 5),
    ):
        try:
            prompt = (Path(prompts_dir) / prompt_file).read_text()
            response = send_request(host, port, [{"role": "user", "content": prompt}],
                                    model=model, max_tokens=max_tokens, timeout=timeout)
            text = ""
            if response.get("choices"):
                text = response["choices"][0]["message"].get("content") or ""
            lang, code = extract_code_fence(text)
            graded = grader(code)
            passed = graded["score"] >= pass_at
            err = response.get("error")
        except Exception as e:
            lang, code, graded, passed, err = "", "", {"score": 0, "max": 0, "checks": {}}, False, str(e)
        results["categories"]["coding"]["total"] += 1
        if passed:
            results["categories"]["coding"]["correct"] += 1
        results["details"].append({"id": test_id, "category": "coding", "pass": passed,
                                   "grade": graded, "lang": lang, "code_chars": len(code),
                                   **({"error": err} if err else {})})
        print(f"  [{'PASS' if passed else 'FAIL'}] {test_id}  score={graded['score']}/{graded['max']}", flush=True)
        _incremental_save(host, port, model, results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="BFCL v2 harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds")
    parser.add_argument("--coding-timeout", type=int, default=6000)
    parser.add_argument("--coding-max-tokens", type=int, default=16384,
                        help="Max generation tokens for coding tests (cap below server ctx - prompt)")
    parser.add_argument("--coding-only", action="store_true")
    parser.add_argument("--tool-tests-only", action="store_true")
    args = parser.parse_args()

    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    if args.output:
        global _INCREMENTAL_PATH
        _INCREMENTAL_PATH = str(Path(args.output))

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "categories": {
            "simple": {"total": 0, "correct": 0},
            "moderate_parallel": {"total": 0, "correct": 0},
            "moderate_chained": {"total": 0, "correct": 0},
        },
        "details": [],
    }

    if not args.coding_only:
        print("=" * 60)
        print("SIMPLE (30 tests)")
        print("=" * 60)
        run_flat_tests(args.host, args.port, args.model, SIMPLE_TESTS,
                       "simple", results, timeout=args.timeout)

        print("=" * 60)
        print("MODERATE PARALLEL (12 tests)")
        print("=" * 60)
        run_flat_tests(args.host, args.port, args.model, MODERATE_PARALLEL,
                       "moderate_parallel", results, timeout=args.timeout)

        print("=" * 60)
        print("MODERATE CHAINED (8 tests)")
        print("=" * 60)
        run_chained_tests(args.host, args.port, args.model, MODERATE_CHAINED,
                          results, timeout=args.timeout)

    if not args.tool_tests_only:
        print("=" * 60)
        print("CODING (2 tests)")
        print("=" * 60)
        run_coding_tests(args.host, args.port, args.model, results, prompts_dir,
                         timeout=args.coding_timeout, max_tokens=args.coding_max_tokens)

    # Summary
    print("\n" + "=" * 60)
    print(f"BFCL v2 Results: {args.model}")
    print("=" * 60)
    tool_cats = ("simple", "moderate_parallel", "moderate_chained")
    tool_total = sum(results["categories"][c]["total"] for c in tool_cats)
    tool_correct = sum(results["categories"][c]["correct"] for c in tool_cats)
    if tool_total:
        print(f"Tool tests: {tool_correct}/{tool_total} ({100*tool_correct/tool_total:.1f}%)")
    if results["categories"].get("coding", {}).get("total"):
        c = results["categories"]["coding"]
        print(f"Coding:    {c['correct']}/{c['total']} ({100*c['correct']/c['total']:.1f}%)")
    total = sum(v["total"] for v in results["categories"].values())
    correct = sum(v["correct"] for v in results["categories"].values())
    if total:
        print(f"Overall:   {correct}/{total} ({100*correct/total:.1f}%)")
        for cat in ("simple", "moderate_parallel", "moderate_chained", "coding"):
            c = results["categories"].get(cat)
            if c and c["total"]:
                print(f"  {cat:<20} {c['correct']:>3}/{c['total']:<3} ({100*c['correct']/c['total']:5.1f}%)")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
