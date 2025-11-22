# LLM Observability (minimal Python SDK)

A tiny, extensible observability layer for LLM calls. Add three lines around your code and get JSON traces for requests, responses, timings, and errors.

## Quick Install

This repo bundles an example using OpenAI.

- Install the package from source (with OpenAI + example extras):
  - `pip install -e .[openai,example]`
- Provide credentials in `.env` (at repo root or in `example/`):
  - `OPENAI_API_KEY=your_key_here`
  - optional: `OPENAI_MODEL=gpt-4o-mini`

Using the SDK in your own project:
- `pip install .` installs the core (no provider deps).
- `pip install .[openai]` adds OpenAI support.

## Quick Start

```python
from llm_observability import observer

observer.observe()    # start a session and auto-instrument providers
# ... make your LLM calls (e.g., OpenAI Chat Completions) ...
observer.end()        # end the session
observer.flush()      # write a single JSON file to disk
```

By default, events flush to `./llm_observability.json`. Override with `LLM_OBS_OUT=/path/to/file.json`.

## Run the Example

- Simple single-file example:
  - `python example/simple-chat-completion/chat.py`
  - Prints the model’s reply and writes events to `llm_observability.json` in the repo root.

- Multi-file pipeline example:
  - `python -m example.pipeline.main "Explain vector databases to a backend engineer"`
  - Runs a 3-step pipeline (research -> summarize -> critique) with multiple Chat Completions calls chained together.

## What gets captured

- Provider: `openai` (Chat Completions v1)
- Request: model, first few `messages`, core params
- Response: text, model, token usage (when available)
- Timing: start/end timestamps, `duration_ms`
- Errors: exception name and message if the call fails
- Callsite: file path, line number, and function name where the API was called

## Data Models

Internally, the SDK structures data with Pydantic models (v2):

- `llm_observability.models.Session`
- `llm_observability.models.Event`
- `llm_observability.models.ObservedEvent` (Event + `session_id`)
- `llm_observability.models.ObservabilityExport` (flush payload)

These are exported to allow downstream tooling to parse and validate the JSON output and to build integrations.

## Extensibility

Providers are classes that implement a small abstract interface and install their own hooks.

- Base class: `llm_observability.providers.base.BaseProvider`
- Built-in: `OpenAIProvider` (auto-detected and installed if `openai` is available)

Custom provider skeleton:

```python
from llm_observability import BaseProvider

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
from llm_observability import observer
observer.register_provider(MyProvider())
observer.observe()
```

If you don’t explicitly register providers, the collector auto-loads built-ins (OpenAI) when `observe()` is called.

## Architecture

- Core
  - `Collector` holds sessions/events and flushes a single JSON file.
  - `llm_observability.models.*` define Pydantic schemas for sessions/events/export.
- Providers (N-layered)
  - `providers/base.py`: `BaseProvider` interface.
  - `providers/openai/provider.py`: orchestrates OpenAI API modules.
  - `providers/openai/apis/base_api.py`: API module interface.
  - `providers/openai/apis/chat_completions.py`: instruments `chat.completions.create`.
  - `providers/openai/apis/models/*`: Pydantic request/response schemas per API.

Providers construct Pydantic request/response models and pass typed `Event` objects to the collector; only the collector serializes to JSON.

## Docs

Sphinx documentation lives under `docs/`.

- Install docs deps (note the quotes for zsh):
  - `pip install '.[docs]'`
- Build HTML docs:
  - `python -m sphinx -b html docs docs/_build/html`
- Open `docs/_build/html/index.html` in your browser.

GitHub Pages
- Docs auto-deploy from `main` via GitHub Actions (pages.yml).
- After merging to `main`, the site is available at:
  - `https://neuralis.github.io/llm-observability/`
  - If your org/user or repo name differs, GitHub Pages uses `https://<owner>.github.io/<repo>/`.
