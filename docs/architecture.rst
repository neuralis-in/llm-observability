Architecture
============

Core
----

- ``Collector`` manages sessions, events, and flushing JSON.
- ``aiobs.models`` provide Pydantic v2 schemas:
  ``Session``, ``Event``, ``FunctionEvent``, ``ObservedEvent``, and ``ObservabilityExport``.

Providers
---------

- Base provider interface: ``aiobs.providers.base.BaseProvider``

**OpenAI Provider** (N-layered):

- ``providers/openai/provider.py``: orchestrates API modules.
- ``providers/openai/apis/base_api.py``: base for API modules.
- ``providers/openai/apis/chat_completions.py``: instruments ``chat.completions.create``.
- ``providers/openai/apis/models/*``: Pydantic models for per-API request/response.

**Gemini Provider** (N-layered):

- ``providers/gemini/provider.py``: orchestrates API modules.
- ``providers/gemini/apis/base_api.py``: base for API modules.
- ``providers/gemini/apis/generate_content.py``: instruments ``models.generate_content``.
- ``providers/gemini/apis/generate_videos.py``: instruments ``models.generate_videos`` (Veo video generation).
- ``providers/gemini/apis/models/*``: Pydantic models for per-API request/response.

Flow
----

1. Call ``observer.observe()`` to start a session and install providers.
2. Make LLM API calls (e.g., OpenAI Chat Completions, Gemini Generate Content, Gemini Generate Videos).
3. Providers build typed request/response models and record an ``Event`` with timing and callsite.
4. ``observer.flush()`` serializes an ``ObservabilityExport`` JSON file.

Trace Tree
----------

Events are linked via ``span_id`` and ``parent_span_id`` fields:

- Each decorated function (``@observe``) generates a unique ``span_id``.
- Nested calls inherit the parent's ``span_id`` as their ``parent_span_id``.
- The ``trace_tree`` field in the export provides a nested view of the execution.

Example trace tree structure::

    {
      "trace_tree": [
        {
          "name": "research",
          "span_id": "abc-123",
          "children": [
            {
              "provider": "openai",
              "api": "chat.completions",
              "parent_span_id": "abc-123"
            }
          ]
        }
      ]
    }

Enhanced Prompt Traces
----------------------

Functions decorated with ``@observe(enh_prompt=True)`` are tracked separately for prompt analysis:

- Each call generates a unique ``enh_prompt_id`` (UUID).
- The ``auto_enhance_after`` parameter specifies how many traces before auto-enhancement runs.
- The ``enh_prompt_traces`` field in the export contains a list of all ``enh_prompt_id`` values.

Example with enhanced prompt tracing::

    @observe(enh_prompt=True, auto_enhance_after=10)
    def summarize(text: str) -> str:
        response = client.chat.completions.create(...)
        return response.choices[0].message.content

The JSON export will include::

    {
      "function_events": [
        {
          "name": "summarize",
          "enh_prompt": true,
          "enh_prompt_id": "bd089fd9-7d25-46df-8a6f-028cf06410f7",
          "auto_enhance_after": 10,
          ...
        }
      ],
      "trace_tree": [
        {
          "name": "summarize",
          "enh_prompt_id": "bd089fd9-7d25-46df-8a6f-028cf06410f7",
          "children": [...]
        }
      ],
      "enh_prompt_traces": [
        "bd089fd9-7d25-46df-8a6f-028cf06410f7"
      ]
    }

This structure allows collecting and analyzing enhanced prompt traces across multiple JSON files.

Extending aiobs
---------------

Create custom providers by implementing ``BaseProvider``::

    from aiobs import BaseProvider

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
            # monkeypatch or add hooks into your SDK
            # call collector._record_event({ ... normalized payload ... })
            def unpatch():
                pass
            return unpatch

    # Register before observe()
    from aiobs import observer
    observer.register_provider(MyProvider())
    observer.observe()
