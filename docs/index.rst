aiobs
=====

Minimal, extensible observability for LLM calls with three lines of code.

.. raw:: html

   <div class="hero">
     <p class="tagline">Observe requests, responses, timings, and errors for your LLM providers. Typed models, pluggable providers, single JSON export.</p>
   </div>

Supported Providers
-------------------

- **OpenAI** — Chat Completions API (``openai>=1.0``)
- **Google Gemini** — Generate Content API (``google-genai>=1.0``)

Classifiers
-----------

Evaluate model response quality with built-in classifiers:

- **OpenAIClassifier** — Uses OpenAI models to determine if responses are good, bad, or uncertain

Evals
-----

Comprehensive evaluation framework for LLM outputs:

- **RegexAssertion** — Check output matches regex patterns
- **SchemaAssertion** — Validate JSON output against JSON Schema
- **GroundTruthEval** — Compare output to expected ground truth
- **HallucinationDetectionEval** — Detect hallucinations using LLM-as-judge
- **LatencyConsistencyEval** — Check latency statistics
- **PIIDetectionEval** — Detect PII leakage in outputs

API Key
-------

An API key is required to use aiobs. Get your free API key from:

👉 **https://neuralis-in.github.io/shepherd/api-keys**

Set it as an environment variable: ``export AIOBS_API_KEY=aiobs_sk_your_key_here``

Quick Start
-----------

.. code-block:: python

   from aiobs import observer

   observer.observe()    # start a session and auto-instrument providers
   # ... make your LLM calls ...
   observer.end()
   observer.flush()      # writes llm_observability.json

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Guide

   getting_started
   usage
   classifier
   evals
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
