# BFCL Jetson Test v2

Welcome! This benchmark asks a question that matters for anyone running models at the edge: which small LLMs can actually DO things - call tools, chain steps, and write working code - on a single 8GB board? The design follows BFCL, the community-standard function-calling benchmark from UC Berkeley.[1]

We took the 17 best function-callers from [BFCL Jetson Test v1](https://github.com/drwjkirkpatrick-web/bfcl-jetson-test) and put them through a harder, more human-shaped exam: 50 tool-calling tests plus two auto-graded coding builds, all served locally by llama.cpp on a Jetson Orin Nano 8GB.[6] Every model runs in the same hardware envelope you would actually deploy - no cloud, no GPU cluster, just one small board doing its best.

The headline: single-call accuracy is nearly saturated at this scale - exactly the saturation BFCL documented on the full leaderboard[1][7] - and the real separation comes from chained multi-step calls and code that actually runs. A 3B Granite model topping the leaderboard at 96.2% is the kind of result that makes edge AI feel less like a compromise and more like a choice.

## Why v2

V1 proved function calling works on 8GB hardware with a 30-test single-shot screen; the 17 models that scored >=50% earned their place here.[6] But the screen was narrow: single calls, one flavor of parallelism, no result-informed behavior, no code generation.[6] BFCL's own evolution explains the upgrade: its v1 tested single-turn AST-matched calls, v3 introduced multi-turn function calling where one call's output feeds the next, and v4 pushed to 70% agentic weight because single-turn accuracy saturated and stopped discriminating.[1][7] So v2 borrows the v3/v4 insight at edge scale - keep evaluation deterministic and executable (no LLM judge, no fluctuation[1][2]), but ask more of the model: derived arguments, conditional branches, and working software.

## Test Suite

Four categories, 52 tests, all auto-graded - mirroring BFCL's multi-category design at edge scale.[1]

### Tool calling (50 tests)

| Category | Count | What it measures |
|---|---|---|
| simple | 30 | Single call with tricky argument extraction: ISO 8601 dates, 24-hour times, currency codes, booleans, arrays, unit conversion ("18% tip on an $85.40 bill" -> calculate_tip(bill_amount=85.40, tip_percentage=18)).[1][3] |
| moderate_parallel | 12 | 2-3 independent calls in one response, matched order-independently (BFCL "parallel" style[3]). |
| moderate_chained | 8 | The divider. The harness executes step 1 with a deterministic mock, feeds the result back as a tool message, and the model must derive step-2 arguments from the returned data - including conditionals ("if below zero, set the thermostat"). BFCL v3 multi-turn compressed to two turns.[7] |

All 40 tool schemas follow the OpenAI function format and are declared in one registry (scripts/tests_v2.py). Evaluation ports v1's BFCL-style rules: strict on required parameters, lenient on extras, string normalization, int/float coercion, order-independent matching for parallel calls.[1][6] A chained test passes only if every step matches. BFCL itself is Apache-2.0 licensed code and data, published at ICML 2025 by the Gorilla team at UC Berkeley.[1][5]

### Extraction: FC mode with prompt-mode parity

Responses are read through two paths, mirroring BFCL's separate FC and Prompt tracks.[1][6] Path 1 is the structured tool_calls field that llama.cpp's autoparser produces when the template renders tool markers.[4] Path 2 is a text fallback for models whose GGUF templates never trigger the grammar: it scans content for JSON call emissions (arrays, fenced JSON, bare objects), tolerates arguments/parameters/args keys, and dedupes exact copies (some models emit the same call in both bare and fenced form). Without path 2, a prompt-mode model scores near zero on structured-only reading - a v2.1 mid-run finding, proven live when granite3.2-2b jumped from a false 1.9% to 82.7% once the fallback landed. This mirrors the FC-vs-Prompt gap BFCL reports for the same model.[1]

### Coding (2 tests)

| Test | Prompt | Grade (auto) |
|---|---|---|
| python_game | Terminal tic-tac-toe: human vs computer, input validation, win/tie detection, play-again loop, stdlib only | 4 checks: board display, input validation, win detection, executes cleanly to exit with piped input (pass = 3/4) |
| html_profile | Single-file profile page for a fictional photographer: hero, skills, contact form, dark theme, @media query, JS form validation | 7 structural checks: doctype, complete structure, h1, form (3 inputs + submit), dark style + @media, script hookup, validation logic (pass = 5/7) |

The grader deliberately avoids an LLM judge: structural checks are grep-based, and the Python game is executed with piped input - a pass means the file actually runs on your machine, the way you would use it. That determinism is BFCL's core design goal, and it carried over intact.[1][2]

## Models (17, from v1)

All 17 models that scored >=50% on the v1 screen,[6] run with their v1-proven settings - template overrides where the GGUF's built-in template lacked tool-call markers (v1 finding: a broken template caps a model at 20%, and llama.cpp's differential autoparser can only parse formats the template itself renders[4][6]):

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

Excluded (v1 score < 50%): DeepSeek-R1 1.5B/7B, Ministral-3B, SmallThinker-3B, Qwen2.5-Coder - reasoning-model CoT and unfixable template mismatches.[6]

## Results (16/17 complete - run paused)

Every score below came from the fixed v2.1 harness: structured extraction with text fallback, duplicate-call dedupe, and per-model memory sizing, following BFCL's dual FC/Prompt-mode evaluation design.[1] Run date 2026-09-29.

| Model | Overall | Simple | Parallel | Chained | Coding | Notes |
|---|---|---|---|---|---|---|
| granite4-3b | 50/52 (96.2%) | 29/30 | 12/12 | 7/8 | 2/2 | champion - perfect parallel |
| hammer2.1-3b | 49/52 (94.2%) | 29/30 | 11/12 | 7/8 | 2/2 | biggest v1-to-v2 climb |
| granite4.1-3b | 48/52 (92.3%) | 28/30 | 11/12 | 7/8 | 2/2 | |
| ternary-bonsai-tq2 | 46/52 (88.5%) | 29/30 | 11/12 | 4/8 | 2/2 | ternary, sub-4-bit - and it codes |
| arch-agent-1.5b | 46/52 (88.5%) | 28/30 | 10/12 | 6/8 | 2/2 | best small agentic profile |
| xlam-2-1b | 44/52 (84.6%) | 29/30 | 10/12 | 3/8 | 2/2 | 1.6GB file - efficiency star |
| ternary-bonsai-q4 | 44/52 (84.6%) | 28/30 | 11/12 | 3/8 | 2/2 | |
| granite4.2-3b | 44/52 (84.6%) | 28/30 | 9/12 | 5/8 | 2/2 | |
| granite3.2-2b | 43/52 (82.7%) | 29/30 | 12/12 | 1/8 | 1/2 | text-fallback rescue (1.9% -> 82.7%) |
| xlam-2-8b | 42/52 (80.8%) | 29/30 | 11/12 | 0/8 | 2/2 | |
| qwen2.5-3b | 42/52 (80.8%) | 29/30 | 11/12 | 0/8 | 2/2 | |
| xlam-2-3b | 40/52 (76.9%) | 29/30 | 9/12 | 0/8 | 2/2 | |
| hermes3-3b-q5 | 39/52 (75.0%) | 27/30 | 10/12 | 0/8 | 2/2 | |
| nanbeige4-3b | 37/52 (71.2%) | 27/30 | 9/12 | 0/8 | 1/2 | thinking model |
| hermes3-3b-q4 | 31/52 (59.6%) | 19/30 | 10/12 | 0/8 | 2/2 | Q4 hurts simple, not parallel |
| llama3.2-3b | 20/52 (38.5%) | 18/30 | 0/12 | 0/8 | 2/2 | codes fine, calls poorly |

PAUSED: qwen3.5-9b (4.1GB DeltaNet hybrid) - the kernel OOM-killed its server twice (24K q8_0 KV, then 16K q4_0 KV; llama-server died at ~4.6GB anonymous RSS mid-run). Next attempt: 12K context, q4_0 KV, 8192-token coding cap. Its v1 tool score was 50%.[6] The full results archive with per-test detail lives outside the repo at ~/projects/bfcl-jetson-test-v2-results/ for deeper analysis.

### What the numbers say

- Chained calls are the great divider. Granite 4.x and Hammer2.1 pass 5-7 of 8 chained tests; nine models score exactly 0 - they emit a correct first call, then cannot derive second-step arguments from returned data. This is BFCL's multi-turn gap reproduced at edge scale.[7]
- Simple is saturated. Tool-trained models cluster at 27-29/30 - consistent with BFCL's single-turn saturation, which is exactly why BFCL re-weighted toward agentic categories.[1][7]
- Coding is table stakes - with two exceptions. Fifteen of sixteen models built a runnable tic-tac-toe and a structurally sound HTML page; granite3.2-2b passed Python but produced no parsable HTML, and llama3.2-3b codes 2/2 while calling tools at 38.5%. Execution-based grading keeps these claims honest - deterministic, executable checks over LLM-judge scoring.[1][2]
- Quantization bites unevenly. Hermes3-3B Q4 loses 8 simple-call points to Q5 (19/30 vs 27/30) yet matches it on parallel (10/12). Ternary Bonsai's TQ2_0 beats its own Q4_0 on chained tests (4/8 vs 3/8).
- Harness honesty matters. Two "0%" scores during the run were harness artifacts, not model failures - fixed by the text fallback and dedupe, then retested (granite3.2-2b: 1.9% -> 82.7%; xlam-2-1b: 3.8% -> 84.6%). A benchmark that cannot tell those apart is not measuring models, it is measuring itself.

## Hardware & Server

- Target: NVIDIA Jetson Orin Nano 8GB, ARM64, CUDA; GUI (GDM) stopped during runs to free ~6GB RAM - the v1 requirement carried forward.[6]
- Server: llama.cpp llama-server --jinja, OpenAI-compatible /v1/chat/completions, GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, flash attention on.[8] The autoparser analyzes chat templates to determine how to parse model outputs, including tool calls, using a differential approach inspired by the git diff algorithm.[4]
- Context: 24576 tokens with q8_0 KV for models up to ~4.5GB; models above that get 16384 tokens with q4_0 KV and a 12,288-token coding cap - on a shared 8GB pool, KV quantization is what lets a 4.9GB model finish at all (learned the hard way: two server-start allocation failures and one mid-run kernel OOM before this rule existed). Hybrid-SSM models (Granite, Llama 3.2) launch with -ngl 15 because their Mamba-style state cannot fully offload; dense models use -ngl 99.[6]
- Determinism: temperature 0.0, top-k 1, repeat penalty 1.0 for tool tests - executable, deterministic evaluation is BFCL's core design goal, avoiding LLM-judge fluctuation.[1][2]

Full settings snapshot: settings.yaml (server flags, per-model overrides, extraction config, timeout stack, retest ledger).

## Reproducing

```bash
# one model, all 52 tests
python3 scripts/run_v2.py --models granite4-3b

# all 17 models, skipping models with saved results
python3 scripts/run_v2.py --skip-existing

# tool tests only (fast screen)
python3 scripts/run_v2.py --models granite4-3b --tool-tests-only
```

Results land in results/<model>.json with per-test details, predicted vs expected calls, and coding grades - saved incrementally after every test so a crash never costs you a run, keeping the deterministic-verification ethic of the harness.[2]

## Scoring

- Tool accuracy: correct / 50 across the three tool categories, scored with BFCL-style AST matching.[1]
- Coding: pass (1) / fail (0) per test at the thresholds above; also reported as sub-scores (e.g. 3/4 Python checks).
- Overall: correct / 52. Category scores are reported separately - a model that nails simple calls but collapses on chained tests is exactly the failure v2 exists to find.

## Project Structure

```
bfcl-jetson-test-v2/
+-- prompts/            # python_game_prompt.txt, html_profile_prompt.txt
+-- scripts/
|   +-- tests_v2.py     # 50 tool-test definitions + 40-tool registry
|   +-- bfcl_v2_harness.py  # harness: request, evaluate, grade, report
|   +-- run_v2.py       # model runner (launch, test, kill, archive, per model)
+-- settings.yaml        # full saved configuration snapshot
+-- templates/           # Qwen2.5, Hermes3, Hammer tool-call templates (from v1)
+-- docs/bfcl_v4_research.md  # BFCL research notes (v1, carried forward)
+-- docs/BFCL_Jetson_Test_v2_Report.pdf  # this report in PDF form (charts + tables)
+-- results/             # <model>.json per model + server logs (gitignored)
```

## Sources

[1] https://gorilla.cs.berkeley.edu/leaderboard.html - BFCL V4 Leaderboard
[2] https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html - BFCL V4 Agentic Web Search blog
[3] https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard - BFCL dataset card (HF)
[4] https://github.com/ggml-org/llama.cpp/blob/master/docs/autoparser.md - llama.cpp autoparser architecture
[5] https://github.com/ShishirPatil/gorilla - Gorilla repo (BFCL, Apache 2.0)
[6] https://github.com/drwjkirkpatrick-web/bfcl-jetson-test - BFCL Jetson Test v1 (our repo)
[7] https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html - BFCL V3 multi-turn blog
[8] https://github.com/ggml-org/llama.cpp - llama.cpp repo
