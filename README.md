# BFCL Jetson Test v2

Robust function-calling and code-generation benchmark for 17 small LLMs on a Jetson Orin Nano 8GB, served by llama.cpp with `--jinja` tool calling. Version 2 of [bfcl-jetson-test](https://github.com/drwjkirkpatrick-web/bfcl-jetson-test): the v1 30-test screen is replaced by a 50-test tool-calling suite plus two auto-graded coding tests.[6]

## Why v2

V1 measured single-shot tool selection on a 30-test screen. The 17 models that passed (≥50%) proved that function calling works on 8GB edge hardware, but the test was narrow: single calls, one category of parallel calls, no result-informed behavior, no code generation.[6] BFCL's own trajectory explains the upgrade: its v1 tested single-turn AST-matched calls, v3 introduced multi-turn and multi-step function calling where the output of one call feeds the next, and v4 pushed the overall score to 70% agentic/multi-turn weight because single-turn accuracy saturated and stopped discriminating.[1][7] This benchmark borrows the v3/v4 insight at edge scale: it keeps deterministic AST-style evaluation (executable, reproducible, no LLM judge[1][2]) but asks more of the model — derived arguments, conditional branches, and working software.

## Test Suite

Four categories, 52 tests, all auto-graded.

### Tool calling (50 tests)

| Category | Count | What it measures |
|---|---|---|
| simple | 30 | Single call with tricky argument extraction: dates in ISO 8601, 24-hour times, currency codes, booleans, arrays, unit conversion ("18% tip on an $85.40 bill" → `calculate_tip(bill_amount=85.40, tip_percentage=18)`).[1][3] |
| moderate_parallel | 12 | 2-3 independent calls in one response, matched order-independently (BFCL "parallel" style[3]). |
| moderate_chained | 8 | Multi-step: the harness executes step 1 with a deterministic mock, feeds the result back as a tool message, and the model must derive step-2 arguments from the returned data — including conditionals ("if below zero, set the thermostat"). This is BFCL v3 multi-turn/multi-step compressed to two turns.[7] |

All 40 tool schemas follow the OpenAI function format and are declared in one registry (`scripts/tests_v2.py`). Evaluation ports v1's BFCL-style rules: strict on required parameters, lenient on extras, string normalization, int/float coercion, order-independent matching for parallel calls.[1][6] A chained test passes only if every step matches. BFCL itself is Apache-2.0 licensed code and data, published at ICML 2025 by the Gorilla team at UC Berkeley.[1][5]

### Extraction: FC mode with prompt-mode parity

Responses are read through two paths, mirroring BFCL's separate FC and Prompt tracks.[1][6] Path 1 is the structured `tool_calls` field that llama.cpp's autoparser produces when the template renders tool markers.[4] Path 2 is a text fallback for models whose GGUF templates never trigger the grammar: it scans content for JSON call emissions (arrays, fenced JSON, bare objects), tolerates `arguments`/`parameters`/`args` keys, and dedupes exact copies (some models emit the same call in both bare and fenced form). Without path 2, a prompt-mode model scores near zero on structured-only reading — a v2.1 mid-run finding (see Retests below). This mirrors the FC-vs-Prompt gap BFCL reports for the same model.[1]

### Coding (2 tests)

| Test | Prompt | Grade (auto) |
|---|---|---|
| python_game | Terminal tic-tac-toe: human vs computer, input validation, win/tie detection, play-again loop, stdlib only | 4 checks: board display, input validation, win detection, executes cleanly to exit with piped input (pass = 3/4) |
| html_profile | Single-file profile page for a fictional photographer: hero, skills, contact form, dark theme, `@media` query, JS form validation | 7 structural checks: doctype, complete structure, h1, form (3 inputs + submit), dark style + @media, script hookup, validation logic (pass = 5/7) |

The coding grader deliberately avoids an LLM judge: structural checks are grep-based, and the Python game is executed with piped input — a pass means the file actually runs. This matches BFCL's design goal of determinism and minimal fluctuation.[1][2]

## Models (17, from v1)

All 17 models that scored ≥50% on the v1 screen,[6] run with their v1-proven settings — template overrides where the GGUF's built-in template lacked tool-call markers (v1 finding: a broken template caps a model at 20%, and llama.cpp's differential autoparser can only parse formats the template itself renders[4][6]):

| Model | Quant | v1 score | Template |
|---|---|---|---|
| Granite 4.0-3B | Q4_K_M | 96.7% | built-in |
| Granite 4.1-3B | Q4_K_M | 93.3% | built-in |
| Granite 4.2-3B | Q4_K_M | 93.3% | built-in |
| Granite 3.2-2B | Q4_K_M | 90.0% | built-in |
| Nanbeige4-3B-Thinking | Q8_0 | 93.3% | built-in |
| Ternary-Bonsai-8B (Q4 lossless) | Q4_0 | 93.3% | built-in |
| Ternary-Bonsai-8B | TQ2_0 | 93.3% | built-in |
| xLAM-2-1B-fc-r | Q8_0 | 93.3% | Qwen2.5 override |
| xLAM-2-3B-fc-r | Q8_0 | 83.3% | Qwen2.5 override |
| xLAM-2-8B-fc-r | Q4_K_M | 90.0% | built-in |
| Arch-Agent-1.5B | Q8_0 | 90.0% | Qwen2.5 override |
| Qwen2.5-3B-Instruct | Q4_K_M | 90.0% | built-in |
| Hermes3-3B (Q5_K_M) | Q5_K_M | 70.0% | Hermes3 override |
| Hermes3-3B (Q4_K_M) | Q4_K_M | 66.7% | Hermes3 override |
| Llama 3.2-3B | Q4_K_M | 56.7% | built-in |
| Hammer2.1-3B | Q8_0 | 53.3% | Hermes3 override |
| Qwen3.5-9B | Q3_K_S | 50.0% | built-in |

Excluded (v1 score < 50%): DeepSeek-R1 1.5B/7B, Ministral-3B, SmallThinker-3B, Qwen2.5-Coder — reasoning-model CoT and unfixable template mismatches.[6]

## Hardware & Server

- **Target:** NVIDIA Jetson Orin Nano 8GB, ARM64, CUDA; GUI (GDM) stopped to free ~6GB RAM — the v1 requirement carried forward.[6]
- **Server:** llama.cpp `llama-server --jinja`, OpenAI-compatible `/v1/chat/completions`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, flash attention on, KV cache q8_0.[8] The autoparser analyzes chat templates to determine how to parse model outputs, including tool calls, using a differential approach inspired by the git diff algorithm.[4]
- **Context:** 24576 tokens with q8_0 KV for every model (coding tests request up to 16384 output tokens; tool tests need far less). Hybrid-SSM models (Granite, Llama 3.2) launch with `-ngl 15` because their Mamba-style state cannot fully offload; dense models use `-ngl 99`.[6]
- **Determinism:** temperature 0.0, top-k 1, repeat penalty 1.0 for tool tests — executable, deterministic evaluation is BFCL's core design goal, avoiding LLM-judge fluctuation.[1][2]

## Reproducing

```bash
# one model, all 52 tests
python3 scripts/run_v2.py --models granite4-3b

# all 17 models, skipping models with saved results
python3 scripts/run_v2.py --skip-existing

# tool tests only (fast screen)
python3 scripts/run_v2.py --models granite4-3b --tool-tests-only
```

Results land in `results/<model>.json` with per-test details, predicted vs expected calls, and coding grades.

## Interim Results (12/17 models, run in progress)

> Captured mid-run 2026-09-29. Two harness fixes landed during the run (text-fallback extraction, duplicate-call dedupe); models completed before the fixes are flagged and queued for retest with the fixed harness. Evaluation tracks BFCL's dual FC/Prompt-mode design, where a model can score very differently depending on how tool definitions are presented.[1]

| Model | Overall | Simple | Parallel | Chained | Coding | Notes |
|---|---|---|---|---|---|---|
| granite4-3b | 50/52 (96.2%) | 29/30 | 12/12 | 7/8 | 2/2 | |
| granite4.1-3b | 48/52 (92.3%) | 28/30 | 11/12 | 7/8 | 2/2 | |
| arch-agent-1.5b | 46/52 (88.5%) | 28/30 | 10/12 | 6/8 | 2/2 | |
| ternary-bonsai-tq2 | 46/52 (88.5%) | 29/30 | 11/12 | 4/8 | 2/2 | |
| granite4.2-3b | 44/52 (84.6%) | 28/30 | 9/12 | 5/8 | 2/2 | |
| qwen2.5-3b | 42/52 (80.8%) | 29/30 | 11/12 | 0/8 | 2/2 | |
| xlam-2-3b | 40/52 (76.9%) | 29/30 | 9/12 | 0/8 | 2/2 | |
| hermes3-3b-q5 | 39/52 (75.0%) | 27/30 | 10/12 | 0/8 | 2/2 | |
| nanbeige4-3b | 37/52 (71.2%) | 27/30 | 9/12 | 0/8 | 1/2 | thinking model |
| hermes3-3b-q4 | 31/52 (59.6%) | 19/30 | 10/12 | 0/8 | 2/2 | |
| xlam-2-1b | 2/52 (3.8%) | 0/30 | 0/12 | 0/8 | 2/2 | harness artifact — retest queued |
| granite3.2-2b | 1/52 (1.9%) | 0/30 | 0/12 | 0/8 | 1/2 | harness artifact — retest queued |

Early signals, pending the 2 retests and 5 remaining models:

- The chained (multi-step) category is the sharpest divider: Granite 4.x passes 5-7 of 8 while most function-calling-trained models score 0 — they emit a correct first call but cannot derive second-step arguments from returned data. This echoes BFCL's finding that multi-turn behavior lags single-turn.[7]
- Simple-call accuracy is nearly saturated among tool-trained models (27-29/30), consistent with BFCL's single-turn saturation, which is why BFCL re-weighted toward agentic categories.[1][7]
- Every completed model except granite3.2-2b passed both coding tests (Python game executed cleanly, HTML structure checks); granite3.2-2b passed Python but returned no parsable HTML. Execution-based grading follows the harness principle of deterministic, executable checks over LLM-judge scoring.[2][1]
- Q4 quantization visibly degrades Hermes3-3B on simple calls (27/30 Q5 vs 19/30 Q4) but not on parallel calls (10/12 both).

## Scoring

- **Tool accuracy:** correct / 50 across the three tool categories.
- **Coding:** pass (1) / fail (0) per test at the thresholds above; also reported as sub-scores (e.g. 3/4 Python checks).
- **Overall:** correct / 52. Category scores are reported separately — a model that nails simple calls but collapses on chained tests is the interesting failure v2 exists to find.

## Project Structure

```
bfcl-jetson-test-v2/
├── prompts/            # python_game_prompt.txt, html_profile_prompt.txt
├── scripts/
│   ├── tests_v2.py     # 50 tool-test definitions + 40-tool registry
│   ├── bfcl_v2_harness.py  # harness: request, evaluate, grade, report
│   └── run_v2.py       # model runner (launch, test, kill, per model)
├── templates/          # Qwen2.5, Hermes3, Hammer tool-call templates (from v1)
├── docs/bfcl_v4_research.md  # BFCL research notes (v1, carried forward)
└── results/            # <model>.json per model + server logs
```

## Sources

[1] https://gorilla.cs.berkeley.edu/leaderboard.html — BFCL V4 Leaderboard
[2] https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html — BFCL V4 Agentic Web Search blog
[3] https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard — BFCL dataset card (HF)
[4] https://github.com/ggml-org/llama.cpp/blob/master/docs/autoparser.md — llama.cpp autoparser architecture
[5] https://github.com/ShishirPatil/gorilla — Gorilla repo (BFCL, Apache 2.0)
[6] https://github.com/drwjkirkpatrick-web/bfcl-jetson-test — BFCL Jetson Test v1 (our repo)
[7] https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html — BFCL V3 multi-turn blog
[8] https://github.com/ggml-org/llama.cpp — llama.cpp repo
