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

```python
import package

# Sync: one‑off
resp = package.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Write a haiku about oceans.",
    config={"temperature": 0.7, "max_tokens": 100},
)

# Batch: returns a handle
client = package.client(provider="openai", model="gpt-4o-mini")
batch = client.submit_batch([
    "Translate to German: Hello world",
    "Translate to German: How are you?",
], batch_options={"display_name": "demo", "completion_window": "24h"})
print(batch.id)
print(batch.status())
# later: results = batch.results()

# Top‑level helper (no client)
batch2 = package.create_batch([
    "Write a meta description for: ...",
    "Write a title for: ...",
], provider="openai", model="gpt-4o-mini", batch_options={"display_name": "seo"})
print(batch2.id)
```

API Overview
------------

- `package.get_response(provider, model, input, config=None, schema=None)` → str or parsed object
- `package.create_batch(inputs, provider, model, config=None, schema=None, batch_options=None)` → `Batch`
- `client = package.client(provider, model, **provider_options)`
  - `client.get_response(input, config=None, schema=None)` → str or parsed object
  - `client.submit_batch(inputs, config=None, schema=None, batch_options=None)` → `Batch`

Configuration
-------------

Use a simple dict or `package.GenerationConfig` with fields:

- `temperature`: float
- `top_p`: float
- `max_tokens`: int
- `seed`: int
- `extra`: dict (provider‑specific params; e.g., `stop`)

Notes:
- OpenAI single calls accept `system` via `config["extra"]["system"]`.
- Gemini batches accept `system_instruction` via `batch_options` or `config.extra`.

Structured Outputs (Remote Only)
--------------------------------

Use `package.StructuredOutput` with either a JSON Schema dict or a Pydantic model.
Supported on OpenAI, Anthropic, and Gemini only. Local providers return plain text.

Example:

```python
from package import StructuredOutput
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    tags: list[str]

# Pydantic
obj = package.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Summarize this text and pick tags: ...",
    schema=StructuredOutput(pydantic_model=Summary),
)

# JSON Schema dict
schema = {"type":"object","properties":{"title":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},"required":["title"],"additionalProperties":False}
data = package.get_response(
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
- Local (`transformers`, `vllm`, `mlx`): Not supported; raises on batch

Batch Options
-------------

Provider‑specific options via `batch_options`:

- OpenAI: `display_name`, `completion_window`, `custom_ids`, `system`
- Gemini: `display_name`, `system_instruction`, `custom_ids`

Batch Examples
--------------

OpenAI: submit, poll, and results
```python
import time
import package
from pydantic import BaseModel

class Item(BaseModel):
    title: str

client = package.client("openai", "gpt-4o-mini")
batch = client.submit_batch(
    inputs=[
        "Write a title for: Why the ocean matters",
        "Write a title for: Building minimal LLM clients",
    ],
    schema=package.StructuredOutput(pydantic_model=Item),
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
import package

client = package.client("gemini", "gemini-1.5-pro")
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
import package

client = package.client("anthropic", "claude-3-5-sonnet-latest")
batch = client.submit_batch(["One", "Two", "Three"])  # Completed handle
print(batch.status())  # "completed"
results = batch.results()  # [str, str, str]
```

Notes & Limitations
-------------------

- OpenAI uses the Responses API exclusively (no legacy fallbacks). For compatible servers, use `OPENAI_BASE_URL`.
- Local providers do not support batch or structured outputs.
- Missing SDKs raise clear `ImportError`s.
