LLMBackend: Minimal, Provider‑Agnostic Text LLM Client
======================================================

Simple, consistent text‑in/text‑out APIs for multiple LLM providers, with
optional structured outputs (remote only) and provider‑native batch jobs.

Highlights
----------

- Single, minimal API for sync and batch text generation
- Remote structured outputs via JSON Schema or Pydantic
- Native batch jobs for OpenAI (Responses Batch) and Gemini (Batches)
- Clean, consistent config across providers (temperature, top_p, max_tokens, seed)

Supported Providers
-------------------

- Remote: `openai` (Responses API only), `anthropic` (Messages API), `gemini` (google‑generativeai)
- Local: `transformers` (Hugging Face), `vllm`, `mlx`

Environment Variables
---------------------

- OpenAI: `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for compatible servers)
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

Quickstart
----------

Installation
------------

- Local checkout: `pip install -e .`
- With provider extras (pick what you need):
  - `pip install -e .[openai]`
  - `pip install -e .[anthropic]`
  - `pip install -e .[gemini]`
  - `pip install -e .[transformers]`
  - `pip install -e .[vllm]`
  - `pip install -e .[mlx]`

```python
import llmbackend

# Sync: one‑off
resp = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Write a haiku about oceans.",
    config={"temperature": 0.7, "max_tokens": 100},
)

# Batch: returns a handle
client = llmbackend.client(provider="openai", model="gpt-4o-mini")
batch = client.submit_batch([
    "Translate to German: Hello world",
    "Translate to German: How are you?",
], batch_options={"display_name": "demo", "completion_window": "24h"})
print(batch.id)
print(batch.status())
# later: results = batch.results()

# Top‑level helper (no client)
batch2 = llmbackend.create_batch([
    "Write a meta description for: ...",
    "Write a title for: ...",
], provider="openai", model="gpt-4o-mini", batch_options={"display_name": "seo"})
print(batch2.id)
```

API Overview
------------

- `llmbackend.get_response(provider, model, input, config=None, schema=None)` → str or parsed object
- `llmbackend.create_batch(inputs, provider, model, config=None, schema=None, batch_options=None)` → `Batch`
- `client = llmbackend.client(provider, model, **provider_options)`
  - `client.get_response(input, config=None, schema=None)` → str or parsed object
  - `client.submit_batch(inputs, config=None, schema=None, batch_options=None)` → `Batch`

Configuration
-------------

Use a simple dict or `llmbackend.GenerationConfig` with fields:

- `temperature`: float
- `top_p`: float
- `max_tokens`: int
- `seed`: int
- `extra`: dict (provider‑specific params; e.g., `stop`)

Notes:
- OpenAI single calls accept `system` via `config["extra"]["system"]`.
- Gemini accepts `system_instruction` via `config["extra"]["system_instruction"]` for sync and batch calls.

Structured Outputs (Remote Only)
--------------------------------

Use `llmbackend.StructuredOutput` with either a JSON Schema dict or a Pydantic model.
Supported on OpenAI, Anthropic, and Gemini only. Local providers return plain text.

Example:

```python
from llmbackend import StructuredOutput
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    tags: list[str]

# Pydantic
obj = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Summarize this text and pick tags: ...",
    schema=StructuredOutput(pydantic_model=Summary),
)

# JSON Schema dict
schema = {"type":"object","properties":{"title":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},"required":["title"],"additionalProperties":False}
data = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Summarize this text and pick tags: ...",
    schema=StructuredOutput(schema=schema),
)
```

Batch Behavior
--------------

- OpenAI: Responses Batch API; handle with `.id`, `.status()`, `.results()`
- Gemini: Batches API; handle with `.id`, `.status()`, `.results()`
- Anthropic: Per‑prompt immediate `CompletedBatch` (no remote job)
- Local (`transformers`): Runs locally (returns `CompletedBatch`)
- Local (`vllm`, `mlx`): Runs locally (returns `CompletedBatch`)

Batch Options
-------------

Provider-specific options via `batch_options`:

- OpenAI: `display_name`, `completion_window`, `custom_ids`, `system`
- Gemini: `display_name`, `custom_ids` (`system_instruction` now comes from `config.extra`)

Batch Examples
--------------

OpenAI: submit, poll, and results
```python
import time
import llmbackend
from pydantic import BaseModel

class Item(BaseModel):
    title: str

client = llmbackend.client("openai", "gpt-4o-mini")
batch = client.submit_batch(
    inputs=[
        "Write a title for: Why the ocean matters",
        "Write a title for: Building minimal LLM clients",
    ],
    schema=llmbackend.StructuredOutput(pydantic_model=Item),
    batch_options={"display_name": "titles", "completion_window": "24h", "custom_ids": ["a", "b"]},
)

while True:
    status = batch.status()
    if status in ("completed", "succeeded", "SUCCEEDED", "COMPLETED"):
        break
    if status in ("failed", "expired", "canceled"):
        raise RuntimeError(f"Batch failed: {status}")
    time.sleep(5)

items = batch.results()  # [Item(...), Item(...)] in custom_id order
```

Gemini: submit, poll, and results
```python
import time
import llmbackend

client = llmbackend.client("gemini", "gemini-1.5-pro")
batch = client.submit_batch(
    inputs=[
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "Summarize: A tale of two cities is a novel...",
    ],
    batch_options={"display_name": "summaries"},
)

while True:
    status = batch.status()
    if status in ("SUCCEEDED", "COMPLETED", "completed"):
        break
    if status in ("FAILED", "CANCELLED"):
        raise RuntimeError(f"Batch failed: {status}")
    time.sleep(5)

summaries = batch.results()  # [str, str]
```

Anthropic: immediate per‑prompt results
```python
import llmbackend

client = llmbackend.client("anthropic", "claude-3-5-sonnet-latest")
batch = client.submit_batch(["One", "Two", "Three"])  # Completed handle
print(batch.status())  # "completed"
results = batch.results()  # [str, str, str]
```

Local Providers
---------------

Transformers: apply chat template + local batch
```python
import llmbackend

client = llmbackend.client(
    "transformers",
    "mistralai/Mistral-7B-Instruct-v0.3",
    pipeline_kwargs={"device_map": "auto"},
    apply_chat_template=True,
)
batch = client.submit_batch(
    [
        "Draft a friendly reminder email.",
        "Suggest a project codename.",
    ],
    config={
        "max_tokens": 128,
        "extra": {
            "conversation": [
                {"role": "system", "content": "You are concise and helpful."},
            ]
        },
    },
)
print(batch.results())  # ["Reminder email ...", "Codename suggestion ..."]
```

vLLM: sampling params via `config.extra`
```python
import llmbackend

client = llmbackend.client("vllm", "facebook/opt-125m")
out = client.get_response(
    "List three futuristic hobbies.",
    config={
        "temperature": 0.8,
        "extra": {"stop": ["\n"], "presence_penalty": 0.5},
    },
)
print(out)
```

MLX: per-prompt conversations in a batch
```python
import llmbackend

client = llmbackend.client(
    "mlx",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    apply_chat_template=True,
)
batch = client.submit_batch(
    ["Einstein story", "Why is the sky blue?"],
    config={
        "extra": {
            "conversations": [
                [{"role": "user", "content": "Write a story about Einstein."}],
                [{"role": "user", "content": "Explain why the sky is blue."}],
            ]
        }
    },
)
print(batch.results())
```

Provider Options (Quick Reference)
----------------------------------

- Transformers
  - `pipeline_kwargs`: forwarded to `transformers.pipeline(...)` (e.g., `device_map`)
  - `model_kwargs`: forwarded under `pipeline(model_kwargs=...)`
  - `tokenizer`: optional tokenizer instance or id
  - `apply_chat_template` (bool), `add_generation_prompt` (bool)

- vLLM
  - `llm_kwargs`: forwarded to `vllm.LLM(...)`
  - Sampling: `config.extra` is passed to `SamplingParams` (e.g., `stop`, penalties)

- MLX
  - `load_kwargs`: forwarded to `mlx_lm.load(...)`
  - `apply_chat_template` (bool), `add_generation_prompt` (bool)

Notes:
- When `conversation` (or `conversations`) is provided, it fully defines the chat input and the raw `prompt` string is not used for that request.

Notes & Limitations
-------------------

- OpenAI uses the Responses API exclusively (no legacy fallbacks). For compatible servers, use `OPENAI_BASE_URL`.
- Local providers do not support structured outputs; batching is available on Transformers, MLX, and vLLM.
- Missing SDKs raise clear `ImportError`s.
- MLX extras:
  - `config.extra["conversation"]` (or `conversations` for per-prompt data) applies the tokenizer chat template before generation.
  - Provider option `apply_chat_template=True` wraps plain prompts in a single `user` message automatically.
- vLLM extras: entries in `config.extra` are forwarded directly to `SamplingParams` (stop strings, penalties, etc.), and local batches return `CompletedBatch`.
- Transformers extras: `config.extra["conversation"]`/`conversations` use the tokenizer's `apply_chat_template`, and provider option `apply_chat_template=True` wraps plain prompts automatically. All other `config.extra` values are passed straight to `pipeline(..., **kwargs)`.
