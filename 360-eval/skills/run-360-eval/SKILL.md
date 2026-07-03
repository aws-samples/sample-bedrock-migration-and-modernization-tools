---
name: run-360-eval
description: >-
  Run LLM evaluations with 360-eval. Use when the user wants to benchmark,
  evaluate, score, or compare one or more LLMs (Amazon Bedrock, OpenAI, Gemini,
  Azure) on a dataset of prompts using LLM-as-a-jury scoring — including quality
  evals (correctness/completeness/format/etc. scored 1-5), latency/throughput
  benchmarks, vision/multimodal evals, multi-turn evals, and Bedrock Advanced
  Prompt Optimization (APO). Covers preparing inputs, running the engine CLI,
  reading results, and generating an HTML report. This skill drives the engine
  programmatically (no web UI needed).
---

# Running evaluations with 360-eval

360-eval benchmarks LLMs on your own dataset. For each prompt it calls each
**target model**, then a panel of **judge models** ("the jury") scores the
response 1–5 on Correctness, Completeness, Relevance, Format, Coherence, and
Following-instructions against a golden answer. It also records latency, tokens,
and cost. You run it via one CLI entrypoint: `src/benchmarks_run.py`.

This skill is for coding/AI agents driving the engine directly. The reference
implementation is the `360-eval` code base.

## When to use this
- "Benchmark / evaluate / compare these models on this dataset"
- "Which model is most accurate / fastest / cheapest for task X"
- "Score these prompt+answer pairs with an LLM judge"
- "Run a vision eval / latency test / prompt-optimization (APO) run"
- "Generate an evaluation report"

---

## 1. Prerequisites (one-time)

1. **Get the code** (the `360-eval` directory of this repo) and install deps:
   ```bash
   cd 360-eval
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **AWS credentials for Bedrock.** The engine authenticates Bedrock models via
   **SigV4 using the AWS default credential chain** — set credentials any standard
   way (`aws configure`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`[/`AWS_SESSION_TOKEN`],
   or SSO) plus a region:
   ```bash
   export AWS_REGION=us-east-1
   aws sts get-caller-identity      # must succeed
   ```
   Enable **Bedrock model access** in that account for the models you'll use
   (AWS Console → Bedrock → Model access). No separate "Bedrock API key" is
   required. *(Optional: litellm also honors `AWS_BEARER_TOKEN_BEDROCK` if you
   prefer key-based Bedrock auth.)*
3. **Other providers (optional):** for non-Bedrock target/judge models, export the
   matching key: `OPENAI_API`, `GOOGLE_API` (Gemini), `AZURE_API_KEY`.
4. **Run commands from the repo root** so internal imports and `config/` resolve.

> Cost/latency: every prompt × invocation × temperature-variation × experiment is
> a real model call. Start with a **small dataset (5–20 rows)** and `parallel_calls`
> tuned to your rate limits.

---

## 2. Prepare the three input files

Put all three in one working directory (e.g. `./run/`).

### a) Scenarios — `scenarios.jsonl` (one JSON object per line)
This is the dataset. Each line:
```json
{"text_prompt": "Summarize: <text>", "expected_output_tokens": 4000, "task": {"task_type": "summarization", "task_criteria": "Faithful, concise summary that matches the golden answer."}, "golden_answer": "<reference answer>", "temperature": 0.2, "user_defined_metrics": "", "structured_output_format": null}
```
Fields:
- `text_prompt` (str, required) — the prompt sent to the target model.
- `golden_answer` (str, required) — the reference the judge compares against.
- `task.task_type` / `task.task_criteria` (str) — what the model should do + what
  "good" means (the judge uses this).
- `temperature` (float) — sampling temperature for the target model.
- `expected_output_tokens` (int) — max output tokens (4000 is a safe default).
- `user_defined_metrics` (str, optional) — comma-separated extra metric names.
- `structured_output_format` (str|null, optional) — `"JSON"`, `"CSV"`, `"YAML"`,
  `"Markdown"`, `"HTML"`, `"XML"` to additionally validate response structure.
- **Vision:** add the image as a top-level field keyed by your image column name,
  e.g. `"url_image": "https://…/page.png"`, and pass `--vision_enabled true`.
- **Criteria instead of a golden answer:** omit `golden_answer` and add
  `"success_criteria": {"must_include": "...", "definition": "...", "must_not_include": "...", "edge_cases": "..."}`.

### b) Target models — `models.jsonl` (one per line)
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-east-1", "input_token_cost": 0.8, "output_token_cost": 3.2}
{"model_id": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "region": "us-east-1", "input_token_cost": 3.0, "output_token_cost": 15.0}
```
- `model_id` — litellm route id. Bedrock: `bedrock/<model>`; OpenAI direct:
  `openai/<model>`; Gemini: `gemini/<model>`; Azure: `azure/<deployment>`.
- `region` — AWS region for Bedrock (use `N/A` for non-Bedrock).
- `input_token_cost` / `output_token_cost` — $ per **1M** tokens (used for cost
  reporting only; set to `0` if you don't care).
- Add **multiple lines to compare models** in one run.
- Optional: `target_rpm` (rate limit), `service_tier`.

### c) Judges — `judges.jsonl` (the jury; same shape as models.jsonl)
```json
{"model_id": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "region": "us-east-1", "input_token_cost": 3.0, "output_token_cost": 15.0}
```
Use 1 strong judge, or several for a multi-judge jury. Prefer a judge from a
different model family than the target to reduce self-evaluation bias.

### Building scenarios from a CSV (prompt + golden columns)
```python
import csv, json
rows = list(csv.DictReader(open("data.csv")))
with open("run/scenarios.jsonl", "w") as f:
    for r in rows[:20]:                       # keep it small while testing
        f.write(json.dumps({
            "text_prompt": r["prompt"],
            "expected_output_tokens": 4000,
            "task": {"task_type": "qa", "task_criteria": "Answer correctly and match the golden answer."},
            "golden_answer": r["golden_answer"],
            "temperature": 0.2,
            "user_defined_metrics": "",
            "structured_output_format": None,
        }) + "\n")
```

---

## 3. Run the evaluation

From the repo root, with `models.jsonl`/`judges.jsonl` next to `scenarios.jsonl`:
```bash
python src/benchmarks_run.py run/scenarios.jsonl \
  --output_dir run/outputs \
  --model_file_name models.jsonl \
  --judge_file_name judges.jsonl \
  --experiment_name my-eval \
  --parallel_calls 4 \
  --invocations_per_scenario 1 \
  --experiment_counts 1 \
  --temperature_variations 1 \
  --evaluation_pass_threshold 3 \
  --stream_evaluation true \
  --report false
```
Notes:
- `--model_file_name` / `--judge_file_name` are resolved **relative to the
  scenarios file's directory** — pass basenames and keep all three together (or
  pass absolute paths).
- `--evaluation_pass_threshold` is an **integer 2–4** (default 3): a metric "passes"
  at score ≥ threshold. (Passing a float like `0.5` errors.)
- `--invocations_per_scenario` × `--temperature_variations` × `--experiment_counts`
  multiply the number of calls per prompt — keep them at 1 for a quick run.
- `--report true` (default) also renders an HTML report at the end (see §5).

---

## 4. Read the results (programmatically)

Outputs land in `--output_dir`:
- `invocations_*.csv` — one row per (model, prompt, invocation).
- `unprocessed/unprocessed_*.json` — attempts that errored (e.g. empty response,
  throttling) with a `reason`.

Key CSV columns: `model_id`, `prompt`, `golden_answer`, `model_response`,
`api_call_status` (`Success` or an error), `input_tokens`, `output_tokens`,
`response_cost`, `time_to_last_byte`, and **`performance_metrics`** (a JSON string).

Parse `performance_metrics` for the verdict and scores:
```python
import glob, json, pandas as pd
df = pd.read_csv(glob.glob("run/outputs/invocations_*.csv")[0])
pm = df["performance_metrics"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

df["passed"] = pm.apply(lambda d: d.get("judge_success") is True)   # overall pass per row
pass_rate = df.groupby("model_id")["passed"].mean()                 # accuracy per model
print(pass_rate)

# per-metric scores (1-5) for a row:
# pm.iloc[0]["judge_details"][0]["scores"] -> {"Correctness":4,"Completeness":3,...}
```
- **Quality / accuracy** per model = mean of `judge_success`.
- **Latency** = `time_to_last_byte`; **cost** = `response_cost`; tokens as columns.
- A non-`Success` `api_call_status` (or a row in `unprocessed/`) is a failed
  attempt, not a quality fail — report it separately.

---

## 5. Generate an HTML report (optional)

Easiest: add `--report true` to the run above — it writes an HTML report (charts +
an AI executive summary) into the output dir. The summary uses a Bedrock model, so
it needs working AWS creds. To generate separately:
```python
import sys; sys.path.insert(0, "src")
from visualization.html_report import create_html_report
create_html_report("run/outputs", "20260101_120000",      # timestamp must be %Y%m%d_%H%M%S
                   evaluation_names=None, model_ids=None,
                   summary_model="bedrock/us.amazon.nova-lite-v1:0", summary_region="us-east-1")
```

---

## 6. Recipes

- **Compare models:** put N lines in `models.jsonl` — all run against the same
  scenarios; compare `pass_rate`, latency, and cost per `model_id`.
- **Latency / throughput only (skip judging, cheaper):** add `--latency_only_mode true`
  and you can omit judges. Read `time_to_first_byte` / `time_to_last_byte` / `throughput_tps`.
- **Vision / multimodal:** include the image field in each scenario
  (`"url_image": "<url>"`) and pass `--vision_enabled true`; use a vision-capable
  target model (e.g. Nova Pro, Claude Sonnet).
- **Multi-turn / multi-shot:** the engine scores prompts **independently**. To
  evaluate a multi-turn dataset, build one `scenarios.jsonl` per turn (using that
  turn's prompt/golden columns) and run each — or concatenate turns as separate
  scenario lines. There is no chained-turn CLI mode.
- **Advanced Prompt Optimization (APO):** optimizes the prompt with a Bedrock APO
  job, then evaluates with the optimized prompt. Requires a **real S3 bucket** and
  AWS APO access, and is slow (~20–50 min per model). Flags:
  ```bash
  --prompt_optimization_mode evaluate_both \   # or: optimize_only
  --eval_id my-apo-001 \
  --apo_bucket <your-real-s3-bucket> \
  --apo_evaluator llmj \                         # or: steering
  --apo_llmj_rubric "Score reply 1-5 on resolution & tone. Return JSON {score,rationale}" \
  --apo_llmj_judge_model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
  # steering instead: --apo_evaluator steering --apo_steering_criteria '["Reply <= 3 sentences","Acknowledge the issue first"]'
  ```
  `evaluate_both` runs original + optimized side-by-side (optimized rows show a
  `_Prompt_Optimized` model id).

---

## 7. Gotchas
- `--evaluation_pass_threshold` must be an **integer (2–4)**, not a ratio.
- Costs scale with `invocations_per_scenario × temperature_variations × experiment_counts × #models` — keep them low for iteration.
- Short-term Bedrock keys are region-scoped; match each model's `region`.
- Self-evaluation (judge same family as target) is flagged but allowed; prefer a
  cross-family judge for unbiased scores.
- `model_response = NaN` / a row in `unprocessed/` = the call failed (often empty
  response or throttle) — surface it, don't count it as a quality result.
- Run from the repo root so `config/` (pricing catalog) and `src/` imports resolve.

## CLI flag reference
| Flag | Default | Purpose |
|---|---|---|
| `input_file` (positional) | — | scenarios JSONL |
| `--output_dir` | `outputs` | where results + report are written |
| `--model_file_name` | — | target models JSONL (rel. to scenarios dir) |
| `--judge_file_name` | — | judges JSONL (rel. to scenarios dir) |
| `--experiment_name` | `Benchmark-<date>` | label for this run |
| `--parallel_calls` | 4 | concurrent model calls |
| `--invocations_per_scenario` | 2 | repeats per prompt |
| `--experiment_counts` | 2 | full-run repeats |
| `--temperature_variations` | 0 | extra temperature samples per prompt |
| `--evaluation_pass_threshold` | 3 | judge score cutoff (int 2–4) |
| `--stream_evaluation` | true | stream target responses |
| `--report` | true | render HTML report after run |
| `--vision_enabled` | false | send images from scenarios |
| `--latency_only_mode` | false | skip judging; latency only |
| `--prompt_optimization_mode` | none | `optimize_only` / `evaluate_both` (APO) |
| `--apo_bucket` / `--eval_id` | — | required for APO (real S3 bucket + id) |
| `--apo_evaluator` | — | `llmj` or `steering` |
| `--apo_llmj_rubric` / `--apo_llmj_judge_model` | — | APO LLM-judge config |
| `--apo_steering_criteria` | — | JSON array of steering rules |
