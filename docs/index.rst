LLM Observability
=================

Minimal, extensible observability for LLM calls with three lines of code.

.. raw:: html

   <div class="hero">
     <p class="tagline">Observe requests, responses, timings, and errors for your LLM providers. Typed models, pluggable providers, single JSON export.</p>
   </div>

Quick Start
-----------

.. code-block:: python

   from llm_observability import observer

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
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
