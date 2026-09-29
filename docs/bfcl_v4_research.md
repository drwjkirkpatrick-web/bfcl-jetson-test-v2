# BFCL v4 (Berkeley Function Calling Leaderboard) — Comprehensive Research

> **Research date:** September 28, 2026
> **Sources:** gorilla.cs.berkeley.edu blogs, GitHub ShishirPatil/gorilla repo, llama.cpp docs, HuggingFace dataset, PyPI bfcl-eval package, third-party analyses

---

## Table of Contents

1. [What Is BFCL v4?](#1-what-is-bfcl-v4)
2. [Test Categories & Sub-Categories](#2-test-categories--sub-categories)
3. [Test Data Format](#3-test-data-format)
4. [Scoring Methodology](#4-scoring-methodology)
5. [Open-Source Test Harness](#5-open-source-test-harness)
6. [API Request/Response Format](#6-api-requestresponse-format)
7. [Total Test Case Counts](#7-total-test-case-counts)
8. [Multi-Turn Category Deep Dive](#8-multi-turn-category-deep-dive)
9. [llama.cpp Native Tool Calling](#9-llamacpp-native-tool-calling)
10. [Adapting BFCL-Style Tests for llama.cpp](#10-adapting-bfcl-style-tests-for-llamacpp)
11. [Quick-Start: Running BFCL Locally](#11-quick-start-running-bfcl-locally)

---

## 1. What Is BFCL v4?

**BFCL (Berkeley Function Calling Leaderboard)** is the first comprehensive, executable benchmark for evaluating LLM function/tool-calling capability. Created by the Gorilla team at UC Berkeley (Shishir Patil, Huanzhi Mao, Fanjia Yan, et al.), published at ICML 2025.

**Version history:**
| Version | Date | Key Addition |
|---------|------|-------------|
| v1 | Feb 2024 | Expert-curated single-turn tests (Simple, Multiple, Parallel), AST evaluation metric |
| v2 | Aug 2024 | Enterprise/OSS-contributed "Live" data (real APIs executed) |
| v3 | Dec 2024 | Multi-turn & multi-step function calling, state-transition evaluation |
| **v4** | Jul 2025 | **Holistic agentic evaluation**: web search, memory management, format sensitivity |

**v4's core shift:** The benchmark moved from pure function-call emission to evaluating *agentic* behavior. The v4 overall score weights:
- **Agentic: 40%** (web search + memory)
- **Multi-Turn: 30%**
- **Live (single-turn, real APIs): 10%**
- **Non-Live (single-turn, AST): 10%**
- **Hallucination/Relevance: 10%**

This means **70% of the v4 score is agentic/multi-turn** — only 20% is classic single-shot function calling. The reason: single-turn accuracy saturated (frontier models cluster in the high 0.9s), so it stopped discriminating.

**Two evaluation tracks for every model:**
- **FC (Function Calling) mode:** Uses the model's native `tools` API parameter (OpenAI-style)
- **Prompt mode:** Function definitions injected into the system prompt as text; model generates tool calls as text output

Scores are reported separately for FC vs Prompt because they can differ significantly for the same model.

**License:** Apache 2.0 (both code and data).

---

## 2. Test Categories & Sub-Categories

### Test Groups (broad categories for CLI selection)

| Group | Description |
|-------|-------------|
| `all` | All categories including non-scoring ones |
| `all_scoring` | All categories that affect the overall score |
| `agentic` | All `memory` and `web_search` categories |
| `multi_turn` | All multi-turn categories |
| `single_turn` | All single-turn categories |
| `live` | User-contributed live test categories |
| `non_live` | Non-user-contributed categories |
| `python` | Python-specific tests |
| `non_python` | Java and JavaScript tests |
| `memory` | Memory-based tests (kv, vector, rec_sum) |
| `web_search` | Web-search test categories |

### Individual Test Categories (24 total)

| Category | Group | Count | Description |
|----------|-------|-------|-------------|
| `simple_python` | Non-Live | 400 | Single Python function call, one function in scope, AST match |
| `simple_java` | Non-Live | 100 | Single Java function call, tests language-specific types (HashMap, etc.) |
| `simple_javascript` | Non-Live | 50 | Single JavaScript function call |
| `multiple` | Non-Live | 200 | 2-4 functions provided; model selects the correct one |
| `parallel` | Non-Live | 200 | One user query → multiple simultaneous function calls |
| `parallel_multiple` | Non-Live | 200 | Combination: multiple functions provided, each called 0+ times in parallel |
| `irrelevance` | Non-Live | 240 | None of the provided functions are relevant; model should decline to call any |
| `live_simple` | Live | 258 | User-contributed simple calls, executed against real APIs |
| `live_multiple` | Live | 1053 | User-contributed multiple-choice calls, executed |
| `live_parallel` | Live | 16 | User-contributed parallel calls, executed |
| `live_parallel_multiple` | Live | 24 | User-contributed parallel+multiple, executed |
| `live_irrelevance` | Live | 882 | User-contributed irrelevance detection, executed |
| `live_relevance` | Live | 18 | User-contributed relevance detection (param values not checked) |
| `multi_turn_base` | Multi-Turn | 200 | Foundational multi-turn; all info provided across turns |
| `multi_turn_miss_func` | Multi-Turn | 200 | Functions withheld; model must recognize and request them |
| `multi_turn_miss_param` | Multi-Turn | 200 | Parameters missing; model must ask user for clarification |
| `multi_turn_long_context` | Multi-Turn | 200 | Extraneous data (hundreds of files/records) to test context management |
| `memory_kv` | Agentic | 155 | Key-value memory store read/write |
| `memory_vector` | Agentic | 155 | Vector database memory store read/write |
| `memory_rec_sum` | Agentic | 155 | Recursive summarization memory store read/write |
| `web_search_base` | Agentic | 100 | Multihop questions with DuckDuckGo search + snippets |
| `web_search_no_snippet` | Agentic | 100 | Same but snippets withheld; model must fetch URL content |
| `format_sensitivity` | Non-Scoring | 5200 | 26 prompt-format variations × 200 test cases (non-scoring) |

### Category Definitions (Semantic Meaning)

- **Simple:** One function in scope → exactly one function call expected. Tests basic tool-call construction.
- **Multiple:** 2-4 functions provided → model selects the correct one. Tests function selection/relevance.
- **Parallel:** One user query → multiple function calls simultaneously (e.g., weather for 3 cities). Tests decomposition.
- **Parallel Multiple:** Multiple functions provided + each called multiple times. Hardest single-turn category.
- **Irrelevance / Relevance:** Tests abstention — model must NOT call any function when none are relevant. "Hallucination measurement."
- **Live:** Real APIs are actually invoked and outputs compared (not just AST). Tests executability.
- **Multi-Turn:** Stateful conversations across multiple user turns; API backend state is tracked.
- **Memory:** Tests reading/writing to persistent memory backends (KV store, vector DB, recursive summarization).
- **Web Search:** Multihop questions requiring iterative DuckDuckGo searches + URL content fetching.
- **Format Sensitivity:** 26 different prompt formatting permutations to test robustness to schema formatting changes (non-scoring — informational only).

---

## 3. Test Data Format

### Dataset Location
- **GitHub:** `ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/`
- **HuggingFace:** `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- Files are JSONL (JSON Lines) — one JSON object per line

### Single-Turn Test Case Format

Each line in a test file (e.g., `BFCL_v4_simple_python.json`) is:

```json
{
  "id": "simple_python_0",
  "question": [{"role": "user", "content": "Calculate the area of a triangle with base 10 and height 5."}],
  "function": [
    {
      "name": "calculate_triangle_area",
      "description": "Calculates the area of a triangle given its base and height.",
      "parameters": {
        "type": "object",
        "properties": {
          "base": {"type": "integer", "description": "The base of the triangle."},
          "height": {"type": "integer", "description": "The height of the triangle."}
        },
        "required": ["base", "height"]
      }
    }
  ]
}
```

Key fields:
- **`id`**: Unique identifier (e.g., `simple_python_0`, `parallel_3`, `multi_turn_base_15`)
- **`question`**: Array of message objects `[{role, content}]`; for single-turn, one user message
- **`function`**: List of tool documentation objects following **OpenAI-style JSON schema** (name, description, parameters.type=object, parameters.properties, required)

### Ground Truth Format (Single-Turn)

Located in `possible_answer/` directory, e.g., `BFCL_v4_simple_python.json`:

```json
{
  "id": "simple_python_0",
  "ground_truth": [
    {"calculate_triangle_area": {"base": [10], "height": [5]}}
  ]
}
```

- Each parameter value is a **list** of acceptable answers (to handle synonyms, e.g., `["New York City", "NYC"]`)
- For parallel calls, multiple entries in the list:

```json
{
  "id": "parallel_0",
  "ground_truth": [
    {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
    {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}}
  ]
}
```

### Multi-Turn Test Case Format

```json
{
  "id": "multi_turn_base_0",
  "question": [
    [{"role": "user", "content": "Go to the document folder and create a temp directory."}],
    [{"role": "user", "content": "Now list all files in the temp directory."}]
  ],
  "initial_config": {"root_dir": "...", "files": [...]},
  "involved_classes": ["GorillaFileSystem"],
  "next": {"func_name": "...", "func_args": {...}}
}
```

- **`question`**: List of lists — each inner list is one turn's user message(s)
- **`initial_config`**: Starting state of the simulated API backend (file system, credentials, database)
- **`involved_classes`**: The simulated API domains needed (gorilla_file_system, trading_bot, vehicle_control, travel_booking, message_api, twitter_api, ticket_api, math_api)

### Multi-Turn Ground Truth Format

```json
{
  "id": "multi_turn_base_0",
  "ground_truth": [
    ["cd(folder='document')", "mkdir(dir_name='temp')"],
    ["cd(folder='temp')", "ls(a=True)"]
  ]
}
```

- Each inner list = expected function calls for that turn
- **Empty list `[]`** = model should NOT make function calls that turn (e.g., ask for clarification in `miss_param` or `miss_func` scenarios)

### Function Documentation (Multi-Turn)

Stored in `multi_turn_func_doc/` as JSONL files grouped by domain:
- `gorilla_file_system.json` — `ls()`, `cd()`, `cat()`, `mkdir()`, `grep()`, etc.
- `math_api.json` — `logarithm()`, `mean()`, `standard_deviation()`
- `trading_bot.json` — `get_stock_info()`, `place_order()`, `get_watchlist()`
- `vehicle_control.json` — `startEngine()`, `displayCarStatus()`, `estimate_distance()`
- `travel_booking.json` — `book_flight()`, `get_nearest_airport_by_city()`, `purchase_insurance()`
- `message_api.json`, `twitter_api.json`, `ticket_api.json`

Each line is one function schema in OpenAI-style JSON format.

---

## 4. Scoring Methodology

### Overall Score Formula

```
Overall Score = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)
```

### Within-Category Averaging

- **Unweighted Average:** Subcategories averaged equally regardless of test count
  - Used for: Agentic, Multi-Turn, Non-Live, Hallucination
  - Example: Agentic = (Web Search score + Memory score) / 2, even though Web Search has 200 entries and Memory has 465
- **Weighted Average:** Subcategories weighted by actual test case counts
  - Used for: Live category
  - Example: Live = (258×live_simple + 1053×live_multiple + 16×live_parallel + 24×live_parallel_multiple + 882×live_irrelevance + 18×live_relevance) / total

### Three Evaluation Methods

#### Method 1: AST (Abstract Syntax Tree) Evaluation — Single-Turn

The model's generated function call is parsed into a Python AST and compared structurally against ground truth. **Deterministic, no LLM judge.**

**Process:**
1. **Parse** the function call string into an AST (e.g., `calculate_triangle_area(base=10, height=5)`)
2. **Function name matching:** Extract and verify function name matches ground truth (dots `.` substituted with `_`)
3. **Required parameter matching:** All parameters listed in `required` must be present in model output
4. **Parameter type & value matching:**
   - `bool`: Exact boolean match; string "true" ≠ boolean `true`
   - `int`/`float`: For Python, `int` accepted for `float` params (auto-conversion); for Java/JS, strict type matching (`5.0` ≠ `5`)
   - `List`/`Tuple`: Order matters, recursive type matching for nested structures
   - `String`: Case-insensitive, whitespace removed, punctuation `,./-_*^` removed
   - `Dict`: Key presence + value accuracy; key order doesn't matter
   - **Optional parameters:** If ground truth has empty string `""`, model can use default or omit; otherwise must provide correct value
   - **Variable parameters:** If parameter value is a variable name, type checking considers both concrete and variable scenarios
5. **Hallucination detection:** Only parameters in the function doc are valid; extra parameters = failure
6. **All-or-nothing:** For multiple/parallel, every model output must match a ground truth answer

**Example ground truth with acceptable variants:**
```json
{"predict_house_price": {"bedrooms": [3], "bathrooms": [2], "area": [1800], "location": ["San Francisco", "San Francisco, CA"]}}
```

#### Method 2: Executable Function Evaluation — Live Category

The model's function call is **actually executed** and the output compared:

- **Non-REST (Python functions):**
  - **Exact match:** Output must exactly match expected result
  - **Real-time match:** For numerical results, within 20% threshold (for live API data)
  - **Structural match:** Output type must match (list length, dict keys)
- **REST API:**
  - **Effective execution:** API call must succeed
  - **Response type accuracy:** Response structure matches (e.g., list of JSON objects)
  - **JSON key consistency:** Key sets match between generated and expected responses

#### Method 3: State-Transition Evaluation — Multi-Turn & Agentic

Instead of comparing function call trajectories, the **final state of the API backend** is checked:

- The API backend (file system, booking system, etc.) is initialized with `initial_config`
- Model executes function calls across turns; each call actually runs against the backend
- After all turns, the backend state is compared to the expected final state
- **Subset-matched:** The model's result is correct if it contains the ground truth as a subset, even with extra function calls or different trajectories
- This handles: inconsistent trajectories, error recovery, redundant actions

**Why state-based over response-based:**
> A model might explore by listing files before proceeding, which isn't wrong but deviates from the expected trajectory. Response-based evaluation marks this as wrong. State-based evaluation only checks the final outcome.

#### Method 4: Exact-Match — Web Search

- Model must format response as `{'answer': '...', 'context': '...'}`
- Only the `answer` field is evaluated
- Normalization: lowercase, remove punctuation `,./-_*^()`
- Exact match against ground-truth phrase after normalization
- If model says "I do not know" → scored as incorrect

---

## 5. Open-Source Test Harness

**Yes, BFCL is fully open-source and can be run locally.**

### Installation

```bash
# Option A: PyPI package (recommended for running without code changes)
pip install bfcl-eval  # NOT "bfcl" — that's an unrelated package!

# Option B: Clone and install editable
conda create -n BFCL python=3.10
conda activate BFCL
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
pip install -e .

# For self-hosted models, add a backend:
pip install -e .[oss_eval_vllm]    # vLLM backend (wider GPU support)
pip install -e .[oss_eval_sglang]  # sglang backend (faster, SM 80+ only)
```

### Running Evaluations

```bash
# Step 1: Generate model responses
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1

# Step 2: Evaluate the generated responses
bfcl evaluate --model MODEL_NAME --test-category TEST_CATEGORY

# Example: test GPT-4o on simple + parallel categories
bfcl generate --model gpt-4o-2024-11-20-FC --test-category simple_python,parallel --num-threads 1
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category simple_python,parallel
```

### Running Against Pre-Existing OpenAI-Compatible Endpoints

This is the critical feature for llama.cpp integration:

```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --skip-server-setup
```

Configure in `.env`:
```env
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8080

# Or for remote deployments:
REMOTE_OPENAI_BASE_URL=https://your-server.com/v1
REMOTE_OPENAI_API_KEY=your-api-key-here
REMOTE_OPENAI_TOKENIZER_PATH=/path/to/local/tokenizer  # Optional
```

When `--skip-server-setup` is used, BFCL sends requests directly to your existing OpenAI-compatible `/v1/chat/completions` endpoint without spinning up its own vLLM/sglang server.

### Selecting Specific Test Cases

```bash
bfcl generate --model MODEL_NAME --run-ids
```

With `test_case_ids_to_generate.json` in project root:
```json
{
    "simple_python": ["simple_python_102", "simple_python_103"],
    "multi_turn_base": ["multi_turn_base_15"]
}
```

### Output Structure

- **Responses:** `result/MODEL_NAME/BFCL_v3_TEST_CATEGORY_result.json`
- **Scores:** `score/MODEL_NAME/BFCL_v3_TEST_CATEGORY_score.json`
- **CSV summaries:** `score/data_overall.csv`, `data_live.csv`, `data_non_live.csv`, `data_multi_turn.csv`

### Adding a New Model

1. Review `bfcl_eval/model_handler/base_handler.py`
2. Implement a new handler class
3. Update `bfcl_eval/constants/model_config.py`
4. Submit a PR

### Supported Models (100+ as of v4)

Includes GPT-4o/5/5.2, Claude Opus 4.5/Sonnet 4.5, Gemini 3 Pro, Grok 4, GLM-4.6, Qwen3 (all sizes), Llama 3.1/3.3/4, DeepSeek V3.2/R1, Mistral Large/Medium/Small, Gorilla OpenFunctions-v2, and many more. Each can be evaluated in FC mode, Prompt mode, or both.

---

## 6. API Request/Response Format

### FC (Function Calling) Mode Request

BFCL uses **OpenAI-compatible tool calling format**. The request sent to the model's API:

```json
{
  "model": "gpt-4o-2024-11-20",
  "messages": [
    {
      "role": "user",
      "content": "Calculate the area of a triangle with base 10 and height 5."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculate_triangle_area",
        "description": "Calculates the area of a triangle given its base and height.",
        "parameters": {
          "type": "object",
          "properties": {
            "base": {"type": "integer", "description": "The base of the triangle."},
            "height": {"type": "integer", "description": "The height of the triangle."}
          },
          "required": ["base", "height"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Expected Response Format (FC Mode)

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "calculate_triangle_area",
              "arguments": "{\"base\": 10, \"height\": 5}"
            }
          }
        ]
      }
    }
  ]
}
```

Key: `finish_reason` is `"tool_calls"`, and `tool_calls[].function.arguments` is a **JSON string** (not a JSON object).

### Prompt Mode Request (for models without native FC)

No `tools` parameter. Instead, function definitions are injected into the system prompt:

```
System: You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the
function, also point it out. You should only return the function call in tools call sections.

User: Questions:Calculate the area of a triangle with base 10 and height 5.
Here is a list of functions in JSON format that you can invoke:
[{"name": "calculate_triangle_area", "description": "...", "parameters": {...}}]
Should you decide to return the function call(s), NO other text MUST be included.
```

### Parallel Calls Response

For parallel categories, the model returns multiple tool calls in one response:

```json
{
  "choices": [{
    "message": {
      "tool_calls": [
        {"function": {"name": "get_weather", "arguments": "{\"location\": \"Boston\"}"}},
        {"function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}},
        {"function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}
      ]
    }
  }]
}
```

### Multi-Turn Flow

1. **Turn 1:** User message + tools → model returns tool_calls
2. **Execute** each tool call against the simulated backend
3. **Feed results back** as `tool` role messages
4. **Turn 2:** Next user message appended → model returns tool_calls
5. Repeat until all turns exhausted
6. **Evaluate** final backend state against ground truth

---

## 7. Total Test Case Counts

### Scoring Test Cases (5,088 total)

| Category | Sub-Category | Count |
|----------|-------------|-------|
| **Agentic (665)** | Web Search (snippet) | 100 |
| | Web Search (no snippet) | 100 |
| | Memory (vector store) | 155 |
| | Memory (key-value store) | 155 |
| | Memory (recursive summarization) | 155 |
| **Multi-Turn (800)** | Base | 200 |
| | Missing Function | 200 |
| | Missing Parameter | 200 |
| | Long Context | 200 |
| **Live (1,351)** | Live Simple | 258 |
| | Live Multiple | 1,053 |
| | Live Parallel | 16 |
| | Live Parallel Multiple | 24 |
| **Non-Live (1,150)** | Simple Python | 400 |
| | Simple Java | 100 |
| | Simple JavaScript | 50 |
| | Multiple | 200 |
| | Parallel | 200 |
| | Parallel Multiple | 200 |
| **Hallucination (1,122)** | Non-Live Irrelevance | 240 |
| | Live Irrelevance | 882 |
| *(Relevance is tracked but listed under Live)* | Live Relevance | 18 |
| **TOTAL SCORING** | | **5,088** |

### Non-Scoring Test Cases (5,218)

| Category | Count |
|----------|-------|
| Format Sensitivity (26 configurations × 200) | 5,200 |
| Relevance (overlap, not separately scored) | 18 |

**Grand total across all files: ~10,306 test entries**

---

## 8. Multi-Turn Category Deep Dive

### Architecture

Multi-turn tests simulate a **stateful API backend** that persists across conversation turns:

1. **Initialization:** Each test entry's `initial_config` sets up the backend state (e.g., pre-existing files, authenticated sessions, database records)
2. **Turn execution:** User message added to history → model generates tool_calls → calls executed against backend → results fed back
3. **State tracking:** The backend tracks all state changes (files created, orders placed, messages sent)
4. **Evaluation:** Final backend state compared to expected state (subset-matched)

### Four Multi-Turn Sub-Categories

#### Base (200 entries)
- All necessary information is provided (in the user message, prior turn results, or via exploration functions)
- The model should handle the interaction without ambiguity
- Tests core multi-turn competency: state retention, chained function calls, cross-turn context

#### Missing Functions (200 entries)
- A subset of necessary functions is **withheld** from the initial function list
- The model must recognize it cannot complete the task and **indicate the issue**
- In the next turn, the missing functions are provided
- Tests: function gap detection, appropriate clarification requests

#### Missing Parameters (200 entries)
- Essential parameters are **missing** from the user request and cannot be inferred from the system
- The model should **request clarification** rather than guessing
- Tests: parameter gap detection, restraint from unwarranted assumptions
- Ground truth for these turns: empty list `[]` (no function calls expected — model should ask instead)

#### Long Context (200 entries)
- Large volumes of extraneous data injected (hundreds of files, thousands of records)
- Tests: information extraction from overwhelming context, maintaining accuracy with long context windows

### API Domains Used

Eight domains, four primary + four cross-functional:

| Primary | Cross-Functional |
|---------|-----------------|
| Gorilla File System (`ls`, `cd`, `cat`, `mkdir`, `grep`) | Message API (`send_message`, `delete_message`) |
| Vehicle Control (`startEngine`, `displayCarStatus`) | Twitter API (`post_tweet`, `retweet`, `comment`) |
| Trading Bot (`get_stock_info`, `place_order`) | Ticket API (`create_ticket`, `get_ticket`, `close_ticket`) |
| Travel Booking (`book_flight`, `get_nearest_airport_by_city`) | Math API (`logarithm`, `mean`, `standard_deviation`) |

### Data Curation Process

1. **API codebase creation** — Custom APIs inspired by real-world use cases
2. **Graph edge construction** — Functions mapped as nodes; edges represent output→input dependencies
3. **Task generation** — Random graph traversal to create execution paths; persona-based question phrasing (from Persona Hub dataset)
4. **Human-labeled ground truth** — Expert labelers create canonical trajectories per turn
5. **Validation** — 11 rounds of filtering; question clarity, executability, alignment, brevity checks; automated `mypy`/`pydocstyle` enforcement

### Evaluation: Subset-Matched State-Based

The key innovation: **the model's trajectory doesn't need to match ground truth exactly**. The final state must contain the ground truth as a subset.

Example:
- **Ground truth:** `get_stock_info(symbol='NVDA')` → `place_order(...)`
- **Model trajectory:** `get_stock_symbol(company='Nvidia')` (fails) → `get_all_stock_symbols()` → pattern match → `get_stock_info(symbol='NVDA')` → `place_order(...)`
- **Result:** ✓ Correct (achieved the goal despite taking more steps)

---

## 9. llama.cpp Native Tool Calling

### Yes, llama.cpp Supports Native Tool Calling

Added in [PR #9639](https://github.com/ggml-org/llama.cpp/pull/9639), llama.cpp's `llama-server` supports **OpenAI-style function calling** via the `/v1/chat/completions` endpoint.

### Requirements

1. **`--jinja` flag** — Required to enable Jinja template processing for tool-aware chat templates
2. **Tool-aware chat template** — The model must have a `chat_template` or `chat_template_tool_use` property that supports tools

### Starting the Server

```bash
# Models with native tool support (no template override needed):
llama-server --jinja -fa -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q6_K_L
llama-server --jinja -fa -hf bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa -hf ibm-granite/granite-4.1-3b-GGUF:Q4_K_M

# Models needing template override:
llama-server --jinja -fa -hf bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja

llama-server --jinja -fa -hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja

llama-server --jinja -fa -hf bartowski/functionary-small-v3.2-GGUF:Q4_K_M \
    --chat-template-file models/templates/meetkai-functionary-medium-v3.2.jinja

# Generic fallback for any model (less efficient, more tokens):
llama-server --jinja -fa -hf bartowski/phi-4-GGUF:Q4_0
llama-server --jinja -fa -hf bartowski/gemma-2-2b-it-GGUF:Q8_0
```

### Supported Native Formats

| Format | Models |
|--------|--------|
| **Llama 3.x** | Llama 3.1, 3.2, 3.3 (including builtin tools: wolfram_alpha, web_search, code_interpreter) |
| **Hermes 2 Pro** | Hermes 2/3, Qwen 2.5 (all variants), many community finetunes |
| **Functionary v3.1/v3.2** | MeetKai Functionary |
| **Mistral Nemo** | Mistral Nemo, Mistral Large 2407, some community finetunes |
| **FireFunction v2** | Fireworks FireFunction |
| **Command R7B** | Cohere Command R7B (with reasoning extraction) |
| **DeepSeek R1** | DeepSeek R1, R1 distills (WIP — reluctant to call tools) |
| **Generic** | Fallback for any model without a recognized template |

### API Request Format (llama.cpp)

Identical to OpenAI format:

```bash
curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "What is the weather in Istanbul?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country/state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }]
}'
```

### Response Format (llama.cpp)

```json
{
  "choices": [
    {
      "finish_reason": "tool",
      "index": 0,
      "message": {
        "content": null,
        "tool_calls": [
          {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"Istanbul\"}"
          }
        ],
        "role": "assistant"
      }
    }
  ],
  "created": 1727287211,
  "model": "gpt-3.5-turbo",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 16,
    "prompt_tokens": 44,
    "total_tokens": 60
  }
}
```

### Parallel Tool Calls

Disabled by default. Enable by passing:
```json
{"parallel_tool_calls": true}
```
in the completion request payload. Supported on some models (not all).

### Key Notes

- **`--jinja` is mandatory** — Without it, tool calling doesn't work
- **Verify template support:** Check `http://localhost:8080/props` for `chat_template` or `chat_template_tool_use`
- **`--chat-template chatml`** can work as a default for many models (YMMV)
- **Avoid extreme KV quantization** (e.g., `-ctk q4_0`) — substantially degrades tool calling performance
- **Generic format** works for any model but consumes more tokens and is less efficient

---

## 10. Adapting BFCL-Style Tests for llama.cpp

### Approach 1: Use BFCL's Built-in OpenAI-Compatible Endpoint Support

BFCL already supports pointing at any OpenAI-compatible endpoint. This is the most straightforward path:

```bash
# 1. Start llama.cpp server with tool-aware template
llama-server --jinja -fa -m /path/to/model.gguf --host 0.0.0.0 --port 8080

# 2. Configure BFCL to use it
export BFCL_PROJECT_ROOT=/path/to/bfcl-workspace
# In .env:
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8080

# 3. Generate responses (skip server setup — use existing llama.cpp server)
bfcl generate --model MODEL_NAME --test-category simple_python,parallel,multiple --skip-server-setup

# 4. Evaluate
bfcl evaluate --model MODEL_NAME --test-category simple_python,parallel,multiple
```

**Challenge:** You need to register your model in BFCL's `model_config.py` or use an existing model handler that sends OpenAI-compatible requests. The `--skip-server-setup` flag tells BFCL to use the existing endpoint instead of spinning up vLLM/sglang.

### Approach 2: Build a Simplified BFCL-Style Test Harness

For a lighter-weight approach that directly tests llama.cpp's `/v1/chat/completions` endpoint:

```python
import json
import requests
import ast
from typing import List, Dict, Any

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"

def load_bfcl_test_cases(filepath: str) -> List[dict]:
    """Load BFCL JSONL test cases."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def load_ground_truth(filepath: str) -> Dict[str, dict]:
    """Load ground truth answers keyed by test ID."""
    truth = {}
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            truth[entry["id"]] = entry["ground_truth"]
    return truth

def send_tool_call_request(question: str, functions: list, model: str = "local-model") -> dict:
    """Send an OpenAI-compatible tool-calling request to llama.cpp."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "tools": [
            {
                "type": "function",
                "function": func
            }
            for func in functions
        ],
        "tool_choice": "auto",
        "temperature": 0.0,  # Deterministic for evaluation
    }
    response = requests.post(LLAMA_CPP_URL, json=payload)
    return response.json()

def extract_tool_calls(response: dict) -> list:
    """Extract tool calls from the OpenAI-format response."""
    message = response["choices"][0]["message"]
    if "tool_calls" not in message or message["tool_calls"] is None:
        return []
    
    calls = []
    for tc in message["tool_calls"]:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        calls.append({"name": name, "arguments": args})
    return calls

def evaluate_ast(predicted: list, ground_truth: list) -> bool:
    """
    Simplified AST-style evaluation.
    Checks: function name match, required parameter presence, value match.
    """
    if len(predicted) != len(ground_truth):
        return False
    
    matched = [False] * len(ground_truth)
    
    for pred in predicted:
        pred_name = pred["name"]
        pred_args = pred["arguments"]
        
        for i, gt_entry in enumerate(ground_truth):
            if matched[i]:
                continue
            
            # ground_truth entry: {"function_name": {"param": [acceptable_values]}}
            for gt_name, gt_params in gt_entry.items():
                if gt_name != pred_name:
                    continue
                
                # Check all parameters match
                all_match = True
                for param_name, acceptable_values in gt_params.items():
                    if param_name not in pred_args:
                        all_match = False
                        break
                    
                    pred_val = str(pred_args[param_name]).strip().lower()
                    pred_val = normalize_string(pred_val)
                    
                    # Check against any acceptable value
                    found = False
                    for acc_val in acceptable_values:
                        acc_normalized = normalize_string(str(acc_val).strip().lower())
                        if pred_val == acc_normalized:
                            found = True
                            break
                    
                    if not found:
                        all_match = False
                        break
                
                if all_match:
                    matched[i] = True
                    break
    
    return all(matched)

def normalize_string(s: str) -> str:
    """Normalize: remove whitespace and punctuation per BFCL rules."""
    import re
    s = s.replace(" ", "")
    s = re.sub(r'[,./\-_*^()]', '', s)
    return s

def run_bfcl_test_category(test_file: str, truth_file: str) -> dict:
    """Run a complete BFCL test category against llama.cpp."""
    test_cases = load_bfcl_test_cases(test_file)
    ground_truth = load_ground_truth(truth_file)
    
    results = {"total": 0, "correct": 0, "details": []}
    
    for tc in test_cases:
        test_id = tc["id"]
        question = tc["question"][0]["content"]  # Single-turn: first user message
        functions = tc["function"]
        
        # Send request to llama.cpp
        try:
            response = send_tool_call_request(question, functions)
            predicted_calls = extract_tool_calls(response)
        except Exception as e:
            predicted_calls = []
            print(f"Error on {test_id}: {e}")
        
        # Evaluate
        gt = ground_truth.get(test_id, [])
        is_correct = evaluate_ast(predicted_calls, gt)
        
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        
        results["details"].append({
            "id": test_id,
            "correct": is_correct,
            "predicted": predicted_calls,
            "expected": gt,
        })
    
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results

# Usage:
# results = run_bfcl_test_category(
#     "BFCL_v4_simple_python.json",
#     "possible_answer/BFCL_v4_simple_python.json"
# )
# print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
```

### Approach 3: Minimal BFCL-Style Test (Self-Contained)

A completely self-contained test that doesn't require downloading BFCL data:

```python
import json
import requests

URL = "http://localhost:8080/v1/chat/completions"

# Define 3 BFCL-style test cases covering Simple, Multiple, and Parallel
TESTS = [
    # 1. Simple: one function, one call
    {
        "id": "simple_1",
        "question": "What is the weather in San Francisco?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "expected": [{"name": "get_weather", "args": {"location": "San Francisco"}}]
    },
    # 2. Multiple: 2 functions, pick the right one
    {
        "id": "multiple_1",
        "question": "Book a flight from NYC to London on March 15",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight between two cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "date": {"type": "string"}
                        },
                        "required": ["origin", "destination", "date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }
        ],
        "expected": [{"name": "book_flight", "args": {"origin": "NYC", "destination": "London", "date": "March 15"}}]
    },
    # 3. Parallel: multiple calls from one query
    {
        "id": "parallel_1",
        "question": "What's the weather in Boston, San Francisco, and Tokyo?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"]
                }
            }
        }],
        "expected": [
            {"name": "get_weather", "args": {"location": "Boston"}},
            {"name": "get_weather", "args": {"location": "San Francisco"}},
            {"name": "get_weather", "args": {"location": "Tokyo"}}
        ]
    },
    # 4. Irrelevance: no function should be called
    {
        "id": "irrelevance_1",
        "question": "Tell me a joke about programming.",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            }
        }],
        "expected": []  # No tool calls expected
    }
]

def run_test(test_case):
    """Run a single BFCL-style test against llama.cpp."""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": test_case["question"]}],
        "tools": test_case["tools"],
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    
    resp = requests.post(URL, json=payload, timeout=30)
    data = resp.json()
    
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    
    predicted = []
    for tc in tool_calls:
        predicted.append({
            "name": tc["function"]["name"],
            "args": json.loads(tc["function"]["arguments"])
        })
    
    # Evaluate
    expected = test_case["expected"]
    
    if len(expected) == 0:
        # Irrelevance: correct if no tool calls
        correct = len(predicted) == 0
    else:
        # Check each expected call is present (order-independent for parallel)
        correct = len(predicted) == len(expected)
        if correct:
            for exp in expected:
                found = any(
                    p["name"] == exp["name"] and 
                    all(str(p["args"].get(k, "")).lower() == str(v).lower() 
                        for k, v in exp["args"].items())
                    for p in predicted
                )
                if not found:
                    correct = False
                    break
    
    return {
        "id": test_case["id"],
        "correct": correct,
        "predicted": predicted,
        "expected": expected,
    }

# Run all tests
if __name__ == "__main__":
    results = [run_test(t) for t in TESTS]
    passed = sum(1 for r in results if r["correct"])
    print(f"\nBFCL-Style Test Results: {passed}/{len(results)} passed\n")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['id']}")
        if not r["correct"]:
            print(f"    Predicted: {r['predicted']}")
            print(f"    Expected:  {r['expected']}")
```

### Key Considerations for llama.cpp + BFCL

| Consideration | Details |
|--------------|---------|
| **Model selection** | Use a model with known tool-calling support: Qwen2.5-7B-Instruct, Hermes-2-Pro, Llama-3.3-70B, Functionary v3.2 |
| **Template** | Must pass `--jinja`; verify tool template at `/props` endpoint |
| **Temperature** | Set to 0.0 for deterministic, reproducible evaluation |
| **Parallel calls** | Add `"parallel_tool_calls": true` to payload for parallel test categories |
| **Irrelevance tests** | The model must return NO tool_calls — test that `message.tool_calls` is null/empty |
| **Multi-turn** | Requires a simulated backend; the simplified harness above only covers single-turn. For multi-turn, use BFCL's full harness with `--skip-server-setup` |
| **Type matching** | BFCL AST evaluation is strict on types (int vs float, bool vs string). Your evaluation must replicate these rules |
| **Java/JS** | llama.cpp's tool calling doesn't natively distinguish Java/JS types. These categories may not work well |
| **Web search** | Requires SerpAPI/DuckDuckGo integration — not suitable for a simplified harness |

---

## 11. Quick-Start: Running BFCL Locally

### Full BFCL Harness Against llama.cpp

```bash
# 1. Install BFCL
pip install bfcl-eval

# 2. Set up workspace
export BFCL_PROJECT_ROOT=~/bfcl-workspace
mkdir -p $BFCL_PROJECT_ROOT
cp $(python -c "import bfcl_eval; print(bfcl_eval.__path__[0])")/.env.example $BFCL_PROJECT_ROOT/.env

# 3. Edit .env — set endpoint to your llama.cpp server
# LOCAL_SERVER_ENDPOINT=localhost
# LOCAL_SERVER_PORT=8080

# 4. Start llama.cpp with tool calling enabled
llama-server --jinja -fa -m /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080

# 5. Run a subset of tests (start with single-turn AST categories)
bfcl generate --model gorilla-openfunctions-v2 \
  --test-category simple_python,parallel,multiple,irrelevance \
  --skip-server-setup

# 6. Evaluate
bfcl evaluate --model gorilla-openfunctions-v2 \
  --test-category simple_python,parallel,multiple,irrelevance

# 7. Check results
cat $BFCL_PROJECT_ROOT/score/data_overall.csv
```

### Minimal Standalone Test (No BFCL Install Needed)

```bash
# Save the "Approach 3" Python script above as bfcl_mini_test.py
# Ensure llama-server is running with --jinja
python bfcl_mini_test.py
```

---

## Summary of Key Findings

| Question | Answer |
|----------|--------|
| **What is BFCL v4?** | Berkeley's function-calling benchmark, v4 adds agentic evaluation (web search, memory, format sensitivity) on top of v1-v3's AST-verified single/multi-turn function calling |
| **Test categories** | 24 individual categories across 5 scoring groups: Non-Live (AST), Live (executable), Multi-Turn (state-based), Agentic (web search + memory), Hallucination (irrelevance) |
| **Test format** | JSONL files; each entry has `id`, `question` (message array), `function` (OpenAI-style JSON schema tool definitions); ground truth in `possible_answer/` |
| **Scoring** | Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%; AST parsing for single-turn, executable for live, state-transition for multi-turn, exact-match for web search |
| **Open-source harness?** | Yes — `pip install bfcl-eval`, full CLI with `generate` and `evaluate` commands, Apache 2.0 |
| **API format** | OpenAI-compatible `tools` parameter in `/v1/chat/completions`; FC mode uses native tool calling, Prompt mode injects function defs into system prompt |
| **Total test cases** | 5,088 scoring + 5,218 non-scoring = ~10,306 total |
| **llama.cpp support?** | Yes — `--jinja` flag enables OpenAI-compatible tool calling; supports Llama 3.x, Hermes 2/3, Qwen 2.5, Mistral Nemo, Functionary, Command R7B, FireFunction, DeepSeek R1, plus generic fallback |
| **Adapt for llama.cpp?** | Two paths: (1) BFCL's `--skip-server-setup` flag points at any OpenAI-compatible endpoint, (2) build a simplified harness using the `/v1/chat/completions` endpoint with `tools` parameter |
| **Multi-turn details** | 800 entries across Base/Missing-Function/Missing-Parameter/Long-Context; uses stateful API backends (file system, trading, travel, vehicle); evaluated by checking final backend state (subset-matched, not trajectory-matched) |

---

## References

- **GitHub repo:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **Live leaderboard:** https://gorilla.cs.berkeley.edu/leaderboard.html
- **PyPI package:** `pip install bfcl-eval` (https://pypi.org/project/bfcl-eval/)
- **HuggingFace dataset:** https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- **BFCL v1 blog (AST methodology):** https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
- **BFCL v3 blog (Multi-Turn):** https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html
- **BFCL v4 blog (Web Search):** https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
- **BFCL v4 blog (Memory):** https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html
- **BFCL v4 blog (Format Sensitivity):** https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
- **TEST_CATEGORIES.md:** https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md
- **SUPPORTED_MODELS.md:** https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/SUPPORTED_MODELS.md
- **llama.cpp function calling docs:** https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- **llama.cpp PR #9639 (tool calling):** https://github.com/ggml-org/llama.cpp/pull/9639
- **ICML 2025 paper:** Patil et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models"
- **Berkeley thesis:** "A Function Calling Perspective on Scalable Large Language Model Agent Evaluation" (EECS-2025-184)