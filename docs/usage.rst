Usage
=====

Simple Chat Completions
-----------------------

The repository includes a simple example at ``example/simple-chat-completion/chat.py``.

Key lines::

    from llm_observability import observer

    observer.observe()
    # Call OpenAI Chat Completions via openai>=1
    observer.end()
    observer.flush()

Pipeline Example
----------------

Chained tasks with multiple API calls::

    python -m example.pipeline.main "Your prompt here"

This runs a three-step pipeline (research → summarize → critique) and writes a single JSON file with all events.

Output
------

By default, ``observer.flush()`` writes ``./llm_observability.json``. Override with the ``LLM_OBS_OUT`` environment variable.

