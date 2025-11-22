Getting Started
===============

Install
-------

- Core only::

    pip install .

- With OpenAI and example extras::

    pip install -e .[openai,example]

Configure
---------

Create a ``.env`` at the repo root (or your project root)::

    OPENAI_API_KEY=your_key_here
    # Optional
    OPENAI_MODEL=gpt-4o-mini

Hello, Observability
--------------------

Minimal example::

    from llm_observability import observer

    observer.observe()
    # ... make your LLM calls ...
    observer.end()
    observer.flush()  # writes llm_observability.json

