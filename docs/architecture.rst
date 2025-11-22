Architecture
============

Core
----

- ``Collector`` manages sessions, events, and flushing JSON.
- ``llm_observability.models`` provide Pydantic v2 schemas:
  ``Session``, ``Event``, ``ObservedEvent``, and ``ObservabilityExport``.

Providers
---------

- Base provider interface: ``llm_observability.providers.base.BaseProvider``
- OpenAI provider (N-layered):
  - ``providers/openai/provider.py``: orchestrates API modules.
  - ``providers/openai/apis/base_api.py``: base for API modules.
  - ``providers/openai/apis/chat_completions.py``: instruments ``chat.completions.create``.
  - ``providers/openai/apis/models/*``: Pydantic models for per-API request/response.

Flow
----

1. Call ``observer.observe()`` to start a session and install providers.
2. Make LLM API calls (e.g., OpenAI Chat Completions).
3. Providers build typed request/response models and record an ``Event`` with timing and callsite.
4. ``observer.flush()`` serializes an ``ObservabilityExport`` JSON file.

