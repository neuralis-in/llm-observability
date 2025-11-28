# aiobs

A tiny, extensible observability layer for LLM calls. Add three lines around your code and get JSON traces for requests, responses, timings, and errors.

## Supported Providers

- **OpenAI** — Chat Completions API (`openai>=1.0`)
- **Google Gemini** — Generate Content API (`google-genai>=1.0`)

## Quick Install

```bash
# Core only
pip install aiobs

# With OpenAI support
pip install aiobs[openai]

# With Gemini support
pip install aiobs[gemini]

# With all providers
pip install aiobs[all]
```

## Get Your API Key

An API key is required to use aiobs. Get your free API key from:

👉 **https://neuralis-in.github.io/shepherd/api-keys**

Once you have your API key, set it as an environment variable:

```bash
export AIOBS_API_KEY=aiobs_sk_your_key_here
```

Or add it to your `.env` file:

```
AIOBS_API_KEY=aiobs_sk_your_key_here
```

## Quick Start

```python
from aiobs import observer

observer.observe()    # start a session and auto-instrument providers
# ... make your LLM calls (OpenAI, Gemini, etc.) ...
observer.end()        # end the session
observer.flush()      # write a single JSON file to disk
```

You can also pass the API key directly:

```python
observer.observe(api_key="aiobs_sk_your_key_here")
```

By default, events flush to `./llm_observability.json`. Override with `LLM_OBS_OUT=/path/to/file.json`.

## Provider Examples

### OpenAI

```python
from aiobs import observer
from openai import OpenAI

observer.observe()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

observer.end()
observer.flush()
```

### Google Gemini

```python
from aiobs import observer
from google import genai

observer.observe()

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello!"
)

observer.end()
observer.flush()
```

## Function Tracing with `@observe`

Trace any function (sync or async) by decorating it with `@observe`:

```python
from aiobs import observer, observe

@observe
def research(query: str) -> list:
    # your logic here
    return results

@observe(name="custom_name")
async def fetch_data(url: str) -> dict:
    # async logic here
    return data

observer.observe(session_name="my-pipeline")
research("What is an API?")
observer.end()
observer.flush()
```

### Decorator Options

| Option | Default | Description |
|--------|---------|-------------|
| `name` | function name | Custom display name for the traced function |
| `capture_args` | `True` | Whether to capture function arguments |
| `capture_result` | `True` | Whether to capture the return value |
| `enh_prompt` | `False` | Mark trace for enhanced prompt analysis |
| `auto_enhance_after` | `None` | Number of traces after which to run auto prompt enhancer |

```python
# Don't capture sensitive arguments
@observe(capture_args=False)
def login(username: str, password: str):
    ...

# Don't capture large return values
@observe(capture_result=False)
def get_large_dataset():
    ...
```

### What Gets Captured

For each decorated function call:
- Function name and module
- Input arguments (args/kwargs)
- Return value
- Timing: start/end timestamps, `duration_ms`
- Errors: exception name and message if the call fails
- Callsite: file path, line number where the function was defined
- Enhanced prompt metadata (`enh_prompt_id`, `auto_enhance_after`) when enabled

### Enhanced Prompt Tracing

Mark functions for automatic prompt enhancement analysis:

```python
from aiobs import observer, observe

@observe(enh_prompt=True, auto_enhance_after=10)
def summarize(text: str) -> str:
    """After 10 traces, auto prompt enhancer will run."""
    response = client.chat.completions.create(...)
    return response.choices[0].message.content

@observe(enh_prompt=True, auto_enhance_after=5)
def analyze(data: dict) -> dict:
    """Different threshold for this function."""
    return process(data)

observer.observe()
summarize("Hello world")
analyze({"key": "value"})
observer.end()
observer.flush()
```

The JSON output will include:
- `enh_prompt_id`: Unique identifier for each enhanced prompt trace
- `auto_enhance_after`: Configured threshold for auto-enhancement
- `enh_prompt_traces`: List of all `enh_prompt_id` values for easy lookup across multiple JSON files

## Run the Examples

- Simple OpenAI example:
  ```bash
  python example/simple-chat-completion/chat.py
  ```

- Gemini example:
  ```bash
  python example/gemini/main.py
  ```

- Multi-file pipeline example:
  ```bash
  python -m example.pipeline.main "Explain vector databases to a backend engineer"
  ```

## What Gets Captured (LLM Calls)

- **Provider**: `openai` or `gemini`
- **API**: e.g., `chat.completions` or `models.generateContent`
- **Request**: model, messages/contents, core parameters
- **Response**: text, model, token usage (when available)
- **Timing**: start/end timestamps, `duration_ms`
- **Errors**: exception name and message if the call fails
- **Callsite**: file path, line number, and function name where the API was called

## Data Models

Internally, the SDK structures data with Pydantic models (v2):

- `aiobs.Session` – Session metadata
- `aiobs.Event` – LLM provider call event
- `aiobs.FunctionEvent` – Decorated function trace event
- `aiobs.ObservedEvent` (Event + `session_id`)
- `aiobs.ObservedFunctionEvent` (FunctionEvent + `session_id`)
- `aiobs.ObservabilityExport` (flush payload)

These are exported to allow downstream tooling to parse and validate the JSON output and to build integrations.

## Extensibility

Providers are classes that implement a small abstract interface and install their own hooks.

- Base class: `aiobs.BaseProvider`
- Built-in: `OpenAIProvider`, `GeminiProvider` (auto-detected and installed if available)

Custom provider skeleton:

```python
from aiobs import BaseProvider, observer

class MyProvider(BaseProvider):
    name = "my-provider"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_sdk  # noqa: F401
            return True
        except Exception:
            return False

    def install(self, collector):
        # monkeypatch or add hooks into your SDK, then
        # call collector._record_event({ ... normalized payload ... })
        def unpatch():
            pass
        return unpatch

# Register before observe()
observer.register_provider(MyProvider())
observer.observe()
```

## Architecture

- **Core**
  - `Collector` holds sessions/events and flushes a single JSON file.
  - `aiobs.models.*` define Pydantic schemas for sessions/events/export.
- **Providers** (N-layered)
  - `providers/base.py`: `BaseProvider` interface.
  - `providers/openai/`: OpenAI Chat Completions instrumentation.
  - `providers/gemini/`: Google Gemini Generate Content instrumentation.

Providers construct Pydantic request/response models and pass typed `Event` objects to the collector; only the collector serializes to JSON.

## Docs

Sphinx documentation lives under `docs/`.

- Install docs deps:
  ```bash
  pip install aiobs[docs]
  ```
- Build HTML docs:
  ```bash
  python -m sphinx -b html docs docs/_build/html
  ```
- Open `docs/_build/html/index.html` in your browser.

**GitHub Pages**
- Docs auto-deploy from `main` via GitHub Actions.
- Site available at: https://neuralis-in.github.io/aiobs/
