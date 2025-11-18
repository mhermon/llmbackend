LLMBackend: Minimal, Provider‑Agnostic LLM Client
=================================================

**Simple, consistent text generation across multiple LLM providers.**

Switch between OpenAI, Anthropic, Gemini, and local models with a single unified API. Support for structured outputs (Pydantic/JSON Schema), native batch jobs, and clean configuration.

**Key Features**
- 🔄 **Provider-agnostic** - Same code works with OpenAI, Anthropic, Gemini, Transformers, vLLM, MLX
- 📦 **Batch processing** - Native batch API support with unified status checking and result retrieval
- 🎯 **Structured outputs** - Type-safe responses via Pydantic models or JSON Schema (remote providers)
- ⚙️ **Consistent config** - temperature, top_p, max_tokens, seed work across all providers
- 🚀 **Minimal & clean** - Simple API, no bloat, easy to swap providers

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Batch Processing](#batch-processing)
- [Structured Outputs](#structured-outputs)
- [Provider-Specific Features](#provider-specific-features)
- [Configuration](#configuration)
- [Examples](#examples)

## Supported Providers

**Remote APIs:**
- `openai` - OpenAI GPT models (Responses API)
- `anthropic` - Anthropic Claude models (Messages API)
- `gemini` - Google Gemini models (generativeai)

**Local Models:**
- `transformers` - Hugging Face Transformers
- `vllm` - vLLM inference engine
- `mlx` - Apple MLX for M1/M2/M3

## Installation

## Installation

**Basic installation:**
```bash
pip install -e .
```

**With provider dependencies** (install only what you need):
```bash
# Remote providers
pip install -e .[openai]      # OpenAI GPT models
pip install -e .[anthropic]   # Anthropic Claude
pip install -e .[gemini]      # Google Gemini

# Local providers  
pip install -e .[transformers]  # Hugging Face
pip install -e .[vllm]          # vLLM
pip install -e .[mlx]           # Apple MLX

# Multiple providers
pip install -e .[openai,anthropic,gemini]
```

**Environment variables** (for remote providers):
```bash
export OPENAI_API_KEY="sk-..."              # Required for OpenAI
export OPENAI_BASE_URL="https://..."       # Optional, for compatible servers
export ANTHROPIC_API_KEY="sk-ant-..."      # Required for Anthropic
export GOOGLE_API_KEY="..."                # Required for Gemini
```

## Quick Start

## Quick Start

### Simple text generation

```python
import llmbackend

# One-off generation
response = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Write a haiku about oceans.",
    config={"temperature": 0.7, "max_tokens": 100},
)
print(response)
```

### Using a client

```python
import llmbackend

# Create a client for reuse
client = llmbackend.client(
    provider="openai",
    model="gpt-4o-mini"
)

# Generate responses
response = client.get_response("Tell me a joke")
print(response)
```

### Batch processing

```python
import llmbackend

client = llmbackend.client(provider="openai", model="gpt-4o-mini")

# Submit batch
batch = client.submit_batch(
    inputs=[
        "Translate to Spanish: Hello",
        "Translate to French: Hello",
        "Translate to German: Hello",
    ],
    batch_options={"display_name": "translations"}
)

# Wait for completion and get results
results = batch.wait().results()
print(results)  # ["Hola", "Bonjour", "Hallo"]
```

### Structured outputs (Pydantic)

```python
import llmbackend
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

schema = llmbackend.StructuredOutput(pydantic_model=MovieReview)

review = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Review the movie Inception",
    schema=schema,
)

print(f"{review.title}: {review.rating}/10")
print(review.summary)
```

## API Reference

### Top-Level Functions

#### `get_response()`
Generate a single response without creating a client.

```python
llmbackend.get_response(
    provider: str,           # "openai", "anthropic", "gemini", etc.
    model: str,              # Model identifier
    input: str,              # Your prompt
    config: dict = None,     # Optional generation config
    schema: StructuredOutput = None,  # Optional structured output
    **provider_options       # Provider-specific options (api_key, etc.)
) -> str | dict
```

**Example:**
```python
response = llmbackend.get_response(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    input="Explain quantum computing",
    config={"max_tokens": 200, "temperature": 0.5}
)
```

#### `client()`
Create a reusable client instance.

```python
llmbackend.client(
    provider: str,
    model: str,
    **provider_options
) -> Client
```

**Example:**
```python
client = llmbackend.client(
    provider="gemini",
    model="gemini-1.5-flash",
    api_key="..."  # Or use GOOGLE_API_KEY env var
)
```

#### `create_batch()`
Submit a batch job without creating a client.

```python
llmbackend.create_batch(
    inputs: list[str],
    provider: str,
    model: str,
    config: dict = None,
    schema: StructuredOutput = None,
    batch_options: dict = None,
    **provider_options
) -> Batch
```

#### `get_batch_status()`
Check the status of a batch job by ID.

```python
llmbackend.get_batch_status(
    batch_id: str,
    provider: str,
    model: str,
    **provider_options
) -> dict
```

**Returns:**
```python
{
    "state": "completed",      # Normalized: pending, in_progress, completed, failed, cancelled
    "provider": "openai",
    "raw_state": "completed",  # Provider-specific state
    "batch_id": "batch_abc123"
}
```

#### `fetch_batch_results()`
Retrieve results from a completed batch by ID.

```python
llmbackend.fetch_batch_results(
    batch_id: str,
    provider: str,
    model: str,
    schema: StructuredOutput = None,
    **provider_options
) -> list
```

### Client Methods

#### `client.get_response()`
```python
client.get_response(
    input: str,
    config: dict = None,
    schema: StructuredOutput = None
) -> str | dict
```

#### `client.submit_batch()`
```python
client.submit_batch(
    inputs: list[str],
    config: dict = None,
    schema: StructuredOutput = None,
    batch_options: dict = None
) -> Batch
```

#### `client.get_batch_status()`
```python
client.get_batch_status(batch_id: str) -> dict
```

#### `client.fetch_batch_results()`
```python
client.fetch_batch_results(
    batch_id: str,
    schema: StructuredOutput = None
) -> list
```

### Batch Handle Methods

When you submit a batch, you get a `Batch` handle with these methods:

#### `batch.id`
Get the batch identifier (property).
```python
batch_id = batch.id  # Save for later retrieval
```

#### `batch.status()`
Get raw provider-specific status.
```python
status = batch.status()  # e.g., "completed", "JOB_STATE_SUCCEEDED", etc.
```

#### `batch.normalized_status()`
Get normalized status string.
```python
status = batch.normalized_status()  # Always one of: pending, in_progress, completed, failed, cancelled
```

#### `batch.results()`
Retrieve results (raises if not completed).
```python
results = batch.results()  # list[str] or list[dict] for structured outputs
```

#### `batch.wait()`
Block until batch completes.
```python
batch.wait(
    poll_interval: float = 5.0,    # Seconds between status checks
    timeout: float = None,          # Max seconds to wait (None = forever)
    callback: callable = None       # Optional progress callback
) -> Batch  # Returns self for chaining
```

**Example:**
```python
# Simple wait
batch.wait()

# With timeout
batch.wait(timeout=300)  # 5 minutes max

# With progress callback
def show_progress(status: str):
    print(f"Status: {status}")

batch.wait(poll_interval=3.0, callback=show_progress)

# Chaining
results = batch.wait().results()
```

## Batch Processing

### Overview

LLMBackend provides unified batch processing across providers:

| Provider | Batch Type | Supports .wait() | Result Retrieval |
|----------|------------|------------------|------------------|
| OpenAI | Async (remote job) | ✅ | ✅ By ID |
| Gemini | Async (remote job) | ✅ | ✅ By ID |
| Anthropic | Immediate (sequential) | ⚡ No-op | ⚡ Immediate |
| Transformers | Immediate (local) | ⚡ No-op | ⚡ Immediate |
| vLLM | Immediate (local) | ⚡ No-op | ⚡ Immediate |
| MLX | Immediate (local) | ⚡ No-op | ⚡ Immediate |

⚡ = Returns `CompletedBatch` immediately (no async job)

### Three Ways to Use Batches

#### 1. Simple: Wait inline
```python
batch = client.submit_batch(prompts)
results = batch.wait().results()
```

#### 2. Manual polling
```python
batch = client.submit_batch(prompts)

while batch.normalized_status() != "completed":
    print(f"Status: {batch.normalized_status()}")
    time.sleep(5)

results = batch.results()
```

#### 3. Retrieve by ID later
```python
# Session 1: Submit and save ID
batch = client.submit_batch(prompts)
batch_id = batch.id
save_to_database(batch_id)

# Session 2: Retrieve later (hours/days later, different machine, etc.)
batch_id = load_from_database()

status = llmbackend.get_batch_status(batch_id, "openai", "gpt-4o-mini")
if status["state"] == "completed":
    results = llmbackend.fetch_batch_results(batch_id, "openai", "gpt-4o-mini")
```

### Batch Status States

All providers normalize to these standard states:

- `"pending"` - Job queued or validating
- `"in_progress"` - Currently processing
- `"completed"` - Successfully finished
- `"failed"` - Processing failed
- `"cancelled"` - Job cancelled or expired

### Batch Options

Configure batch jobs with provider-specific options:

**OpenAI:**
```python
batch_options={
    "display_name": "my-batch",      # Friendly name
    "completion_window": "24h",      # 24h or 48h
    "custom_ids": ["id1", "id2"],    # Custom IDs for ordering
    "system": "You are helpful."     # System prompt
}
```

**Gemini:**
```python
batch_options={
    "display_name": "my-batch",      # Friendly name
    "custom_ids": ["id1", "id2"],    # Custom IDs for ordering
}
# System instructions via config.extra["system"]
```

### Complete Batch Example
### Complete Batch Example

```python
import llmbackend
from pydantic import BaseModel

class Translation(BaseModel):
    original: str
    translated: str
    language: str

# Create client
client = llmbackend.client("openai", "gpt-4o-mini")

# Submit batch with structured outputs
schema = llmbackend.StructuredOutput(pydantic_model=Translation)
batch = client.submit_batch(
    inputs=[
        "Translate to Spanish: Hello, how are you?",
        "Translate to French: Good morning",
    ],
    schema=schema,
    batch_options={
        "display_name": "translations",
        "custom_ids": ["es", "fr"],
        "completion_window": "24h",
    }
)

# Wait with progress
def show_progress(status: str):
    print(f"Batch status: {status}")

batch.wait(poll_interval=5.0, timeout=600, callback=show_progress)

# Get results
translations = batch.results()
for translation in translations:
    print(f"{translation.language}: {translation.translated}")
```

## Structured Outputs

Structured outputs let you get type-safe, validated responses from remote providers (OpenAI, Anthropic, Gemini). **Not supported on local providers.**

### Using Pydantic Models

```python
from pydantic import BaseModel
from llmbackend import StructuredOutput

class RecipeInfo(BaseModel):
    name: str
    prep_time_minutes: int
    ingredients: list[str]
    difficulty: str  # "easy", "medium", "hard"

schema = StructuredOutput(pydantic_model=RecipeInfo)

recipe = llmbackend.get_response(
    provider="openai",
    model="gpt-4o-mini",
    input="Give me a simple pasta recipe",
    schema=schema,
)

# Returns a RecipeInfo instance
print(recipe.name)
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
```

### Using JSON Schema

```python
from llmbackend import StructuredOutput

json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "confidence"],
    "additionalProperties": False
}

schema = StructuredOutput(
    schema=json_schema,
    name="SentimentAnalysis"
)

analysis = llmbackend.get_response(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    input="Analyze sentiment: This product exceeded my expectations!",
    schema=schema,
)

# Returns a dict
print(f"Sentiment: {analysis['sentiment']}")
print(f"Confidence: {analysis['confidence']}")
```

### Custom Parser

```python
from llmbackend import StructuredOutput

def parse_coordinates(data):
    return (data["latitude"], data["longitude"])

schema = StructuredOutput(
    schema={
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"]
    },
    parser=parse_coordinates
)

coords = llmbackend.get_response(
    provider="gemini",
    model="gemini-1.5-flash",
    input="What are the coordinates of Paris?",
    schema=schema,
)

# Returns tuple from custom parser
lat, lon = coords
print(f"Paris: {lat}, {lon}")
```

## Configuration

### Generation Config

Control generation parameters consistently across providers:

```python
config = {
    "temperature": 0.7,        # Randomness (0.0 = deterministic, 1.0+ = creative)
    "top_p": 0.9,              # Nucleus sampling
    "max_tokens": 500,         # Maximum tokens to generate
    "seed": 42,                # Random seed for reproducibility
    "extra": {                 # Provider-specific extras
        "stop": ["\n\n"],      # Stop sequences
        "system": "You are helpful and concise."  # System prompt
    }
}

response = client.get_response("Tell me about AI", config=config)
```

Or use the `GenerationConfig` class:

```python
from llmbackend import GenerationConfig

config = GenerationConfig(
    temperature=0.8,
    max_tokens=200,
    extra={"stop": ["</response>"]}
)
```

### System Instructions

Provide system instructions via `config.extra["system"]`:

```python
config = {
    "extra": {
        "system": "You are a helpful assistant specialized in Python programming."
    }
}

response = client.get_response(
    "How do I read a CSV file?",
    config=config
)
```

For Gemini, you can also use `system_instruction` (alias for compatibility):

```python
config = {
    "extra": {
        "system_instruction": "You are a data scientist."
    }
}
```

## Provider-Specific Features

### OpenAI

**Supported models:** `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.

```python
client = llmbackend.client(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...",           # Or use OPENAI_API_KEY env var
    base_url="https://..."      # Optional: for OpenAI-compatible servers
)

# Batch with custom completion window
batch = client.submit_batch(
    prompts,
    batch_options={
        "completion_window": "24h",  # "24h" or "48h"
        "custom_ids": ["req1", "req2"],
    }
)
```

### Anthropic

**Supported models:** `claude-3-5-sonnet-latest`, `claude-3-opus-latest`, etc.

```python
client = llmbackend.client(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    api_key="sk-ant-..."  # Or use ANTHROPIC_API_KEY env var
)

# Note: Anthropic batches run sequentially (no async job)
batch = client.submit_batch(prompts)
results = batch.results()  # Available immediately
```

### Gemini

**Supported models:** `gemini-1.5-pro`, `gemini-1.5-flash`, etc.

```python
client = llmbackend.client(
    provider="gemini",
    model="gemini-1.5-flash",
    api_key="..."  # Or use GOOGLE_API_KEY env var
)

# System instructions via config
batch = client.submit_batch(
    prompts,
    config={"extra": {"system": "You are a creative writer."}},
    batch_options={"display_name": "stories"}
)
```

### Transformers (Hugging Face)

**Local inference** with Hugging Face Transformers.

```python
client = llmbackend.client(
    provider="transformers",
    model="mistralai/Mistral-7B-Instruct-v0.3",  # HF repo or local path
    pipeline_kwargs={"device_map": "auto"},      # GPU/CPU allocation
    apply_chat_template=True,                    # Auto-wrap in chat format
)

# Use conversations for chat models
response = client.get_response(
    "Write a poem",
    config={
        "max_tokens": 200,
        "extra": {
            "conversation": [
                {"role": "system", "content": "You are a poet."},
                {"role": "user", "content": "Write a poem about the ocean"}
            ]
        }
    }
)

# Batch with per-prompt conversations
batch = client.submit_batch(
    ["Prompt 1", "Prompt 2"],
    config={
        "extra": {
            "conversations": [
                [{"role": "user", "content": "Prompt 1"}],
                [{"role": "user", "content": "Prompt 2"}],
            ]
        }
    }
)
```

**Provider options:**
- `pipeline_kwargs` - Passed to `transformers.pipeline()` (e.g., `device_map`, `torch_dtype`)
- `model_kwargs` - Model-specific kwargs
- `tokenizer` - Custom tokenizer instance or ID
- `apply_chat_template` (bool) - Auto-wrap prompts in chat format
- `add_generation_prompt` (bool) - Add generation prompt (default: True)

### vLLM

**High-performance local inference** with vLLM.

```python
client = llmbackend.client(
    provider="vllm",
    model="facebook/opt-125m",  # Or local path
    llm_kwargs={"tensor_parallel_size": 2}  # vLLM-specific options
)

# Sampling parameters via config.extra
response = client.get_response(
    "Generate creative ideas",
    config={
        "temperature": 0.9,
        "extra": {
            "stop": ["\n\n"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
        }
    }
)
```

**Provider options:**
- `llm_kwargs` - Passed to `vllm.LLM()` constructor
- All `config.extra` entries are passed to `SamplingParams`

### MLX (Apple Silicon)

**Optimized for M1/M2/M3 chips** with Apple MLX.

```python
client = llmbackend.client(
    provider="mlx",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    load_kwargs={},                    # Passed to mlx_lm.load()
    apply_chat_template=True,          # Auto-wrap in chat format
)

# Per-prompt conversations
batch = client.submit_batch(
    ["Question 1", "Question 2"],
    config={
        "max_tokens": 150,
        "extra": {
            "system": "You are concise.",
            "conversations": [
                [{"role": "user", "content": "Explain AI"}],
                [{"role": "user", "content": "Explain ML"}],
            ],
            "sampler": {"temp": 0.7, "top_p": 0.9}  # MLX sampler params
        }
    }
)
```

**Provider options:**
- `load_kwargs` - Passed to `mlx_lm.load()`
- `apply_chat_template` (bool) - Auto-wrap prompts
- `add_generation_prompt` (bool) - Add generation prompt (default: True)

## Examples

### Example 1: Compare Responses Across Providers

```python
import llmbackend

prompt = "Explain photosynthesis in one sentence."
providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-latest"),
    ("gemini", "gemini-1.5-flash"),
]

for provider, model in providers:
    response = llmbackend.get_response(
        provider=provider,
        model=model,
        input=prompt,
        config={"max_tokens": 100, "temperature": 0.5}
    )
    print(f"\n{provider}:")
    print(response)
```

### Example 2: Batch Processing with Error Handling

```python
### Example 2: Batch Processing with Error Handling

```python
import llmbackend

client = llmbackend.client("openai", "gpt-4o-mini")

prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
batch = client.submit_batch(prompts)

try:
    # Wait with timeout
    batch.wait(timeout=300)
    results = batch.results()
    print(f"✓ Got {len(results)} results")
except TimeoutError:
    print(f"⚠ Batch {batch.id} timed out")
    # Save for later retrieval
    save_batch_id(batch.id)
except RuntimeError as e:
    print(f"✗ Batch failed: {e}")
```

### Example 3: Structured Data Extraction at Scale

```python
import llmbackend
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    category: str
    price_range: str
    features: list[str]

# Product descriptions to analyze
descriptions = [
    "iPhone 15 Pro with titanium design, A17 chip, and advanced camera system. $999-1199",
    "Sony WH-1000XM5 wireless headphones with noise cancellation. Premium audio. $399",
    # ... hundreds more
]

client = llmbackend.client("openai", "gpt-4o-mini")
schema = llmbackend.StructuredOutput(pydantic_model=ProductInfo)

batch = client.submit_batch(
    inputs=[f"Extract product info: {desc}" for desc in descriptions],
    schema=schema,
    batch_options={
        "display_name": "product-extraction",
        "custom_ids": [f"prod_{i}" for i in range(len(descriptions))]
    }
)

# Process when ready
products = batch.wait().results()
for product in products:
    print(f"{product.name} - {product.category}: {product.price_range}")
```

### Example 4: Multi-Provider Fallback

```python
import llmbackend

def get_response_with_fallback(prompt, config=None):
    """Try multiple providers with fallback."""
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("gemini", "gemini-1.5-flash"),
    ]
    
    for provider, model in providers:
        try:
            return llmbackend.get_response(
                provider=provider,
                model=model,
                input=prompt,
                config=config
            )
        except Exception as e:
            print(f"✗ {provider} failed: {e}")
            continue
    
    raise RuntimeError("All providers failed")

# Use it
response = get_response_with_fallback(
    "Explain machine learning",
    config={"max_tokens": 200}
)
```

### Example 5: Local Model with Chat Template

```python
import llmbackend

client = llmbackend.client(
    provider="transformers",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    pipeline_kwargs={"device_map": "auto"},
    apply_chat_template=True
)

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I sort a list in Python?"},
]

response = client.get_response(
    "irrelevant",  # Not used when conversation is provided
    config={"extra": {"conversation": conversation}}
)
print(response)
```

## Testing & Development

### Run Provider Examples

Test all providers with real API calls:

```bash
# Test all configured providers
python scripts/run_provider_examples.py --providers all

# Test specific providers
python scripts/run_provider_examples.py --providers openai,gemini

# Test local providers only
python scripts/run_provider_examples.py --providers transformers,vllm,mlx
```

**Required environment variables:**

```bash
# Remote providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Optional: specify models for testing
export LLMBACKEND_TEST_OPENAI_MODEL="gpt-4o-mini"
export LLMBACKEND_TEST_ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
export LLMBACKEND_TEST_GEMINI_MODEL="gemini-1.5-flash"

# Local providers (models)
export LLMBACKEND_TEST_TRANSFORMERS_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export LLMBACKEND_TEST_VLLM_MODEL="/path/to/model"
export LLMBACKEND_TEST_MLX_MODEL="/path/to/mlx/model"
```

Providers without required credentials are automatically skipped.

## Notes & Limitations

**General:**
- Local providers (Transformers, vLLM, MLX) do not support structured outputs
- Missing SDKs raise clear `ImportError` messages
- All providers support batching (async for OpenAI/Gemini, immediate for others)

**OpenAI:**
- Uses Responses API exclusively (no legacy endpoints)
- Set `OPENAI_BASE_URL` for OpenAI-compatible servers
- Batch jobs: 24h or 48h completion windows
- 50,000 request limit per batch

**Anthropic:**
- Batch submissions execute sequentially (no async batch API)
- Returns `CompletedBatch` immediately

**Gemini:**
- System instructions: use `config.extra["system"]` or `config.extra["system_instruction"]`
- Batch results available via inline responses or cloud storage

**Transformers:**
- `config.extra["conversation"]` or `conversations` for chat models
- `apply_chat_template=True` auto-wraps plain prompts
- All other `config.extra` values passed to `pipeline()`

**vLLM:**
- All `config.extra` entries forwarded to `SamplingParams`
- Supports stop sequences, penalties, etc.

**MLX:**
- Optimized for Apple Silicon (M1/M2/M3)
- `config.extra["sampler"]` for sampler parameters
- `conversations` for per-prompt chat histories

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Keep code minimal and clean
2. Maintain provider-agnostic abstractions  
3. Add tests for new features
4. Update documentation

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
