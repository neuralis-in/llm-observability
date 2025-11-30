Usage
=====

Session Labels
--------------

Add labels to sessions for filtering and categorization in enterprise dashboards::

    from aiobs import observer

    observer.observe(
        session_name="my-session",
        labels={
            "environment": "production",
            "team": "ml-platform",
            "project": "recommendation-engine",
            "version": "v2.3.1",
        }
    )

    # ... your LLM calls ...

    observer.end()
    observer.flush()

Labels are key-value string pairs that enable:

- **Dashboard filtering**: Filter sessions by environment, team, project, etc.
- **Cost attribution**: Track usage by team or project
- **Comparison**: Compare metrics across environments (prod vs staging)

Label Constraints
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Constraint
     - Specification
   * - Key format
     - Lowercase alphanumeric with underscores (``^[a-z][a-z0-9_]{0,62}$``)
   * - Value format
     - UTF-8 string, max 256 characters
   * - Max labels
     - 64 per session
   * - Reserved prefix
     - ``aiobs_`` (used for system labels)

Dynamic Label Updates
^^^^^^^^^^^^^^^^^^^^^

Update labels during an active session::

    from aiobs import observer

    observer.observe(labels={"environment": "staging"})

    # Add a single label
    observer.add_label("user_tier", "enterprise")

    # Update multiple labels (merge with existing)
    observer.set_labels({"experiment_id": "exp-42", "feature_flag": "new_model"})

    # Replace all user labels (system labels preserved)
    observer.set_labels({"environment": "production"}, merge=False)

    # Remove a label
    observer.remove_label("experiment_id")

    # Get current labels
    labels = observer.get_labels()
    print(labels)  # {'environment': 'production', 'aiobs_sdk_version': '0.1.0', ...}

    observer.end()
    observer.flush()

Environment Variable Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Labels can be auto-populated from environment variables::

    # Set in shell or .env
    export AIOBS_LABEL_ENVIRONMENT=production
    export AIOBS_LABEL_TEAM=ml-platform
    export AIOBS_LABEL_SERVICE=api-gateway

These are automatically merged with explicit labels (explicit takes precedence)::

    # AIOBS_LABEL_ENVIRONMENT=staging is set in env

    observer.observe(labels={"environment": "production"})
    labels = observer.get_labels()
    print(labels["environment"])  # "production" (explicit overrides env)

System Labels
^^^^^^^^^^^^^

The following labels are automatically added to every session:

- ``aiobs_sdk_version``: SDK version
- ``aiobs_python_version``: Python runtime version
- ``aiobs_hostname``: Machine hostname
- ``aiobs_os``: Operating system

Simple Chat Completions (OpenAI)
--------------------------------

The repository includes a simple example at ``example/simple-chat-completion/chat.py``.

Key lines::

    from aiobs import observer

    observer.observe()
    # Call OpenAI Chat Completions via openai>=1
    observer.end()
    observer.flush()

OpenAI Embeddings
-----------------

aiobs automatically instruments OpenAI's ``embeddings.create`` API::

    from aiobs import observer
    from openai import OpenAI

    observer.observe()

    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello world"
    )

    observer.end()
    observer.flush()

The captured data includes:

- **Request**: model, input text(s), encoding_format, dimensions
- **Response**: embedding vectors, dimensions, usage statistics
- **Timing**: start/end timestamps, ``duration_ms``

For batch embeddings with multiple inputs::

    from aiobs import observer
    from openai import OpenAI

    observer.observe()

    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=["Hello world", "Goodbye world", "How are you?"]
    )

    observer.end()
    observer.flush()

Gemini Generate Content
-----------------------

Example using Google Gemini at ``example/gemini/main.py``.

Key lines::

    from aiobs import observer
    from google import genai

    observer.observe()

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain quantum computing"
    )

    observer.end()
    observer.flush()

Gemini Video Generation (Veo)
-----------------------------

aiobs automatically instruments Google's Veo video generation API (``models.generate_videos``)::

    from aiobs import observer
    from google import genai

    observer.observe()

    client = genai.Client()
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="A cinematic shot of waves crashing on a beach at sunset",
    )

    # Poll until video is ready
    while not operation.done:
        time.sleep(10)
        operation = client.operations.get(operation)

    observer.end()
    observer.flush()

The captured data includes:

- **Request**: model, prompt, image (for image-to-video), video (for video extension), config
- **Response**: operation_name, done status, generated_videos metadata
- **Timing**: start/end timestamps, ``duration_ms``
- **Config options**: aspect_ratio, resolution, number_of_videos, generate_audio, etc.

For image-to-video generation::

    from aiobs import observer
    from google import genai

    observer.observe()

    client = genai.Client()
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="Animate this landscape",
        image=image_object,  # Generated or loaded image
        config={
            "aspect_ratio": "16:9",
            "resolution": "720p",
        }
    )

    observer.end()
    observer.flush()

Function Tracing with @observe
------------------------------

Trace any function (sync or async) with the ``@observe`` decorator::

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

Decorator Options
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``name``
     - function name
     - Custom display name for the traced function
   * - ``capture_args``
     - ``True``
     - Whether to capture function arguments
   * - ``capture_result``
     - ``True``
     - Whether to capture the return value
   * - ``enh_prompt``
     - ``False``
     - Mark trace for enhanced prompt analysis
   * - ``auto_enhance_after``
     - ``None``
     - Number of traces after which to run auto prompt enhancer

Examples::

    # Don't capture sensitive arguments
    @observe(capture_args=False)
    def login(username: str, password: str):
        ...

    # Don't capture large return values
    @observe(capture_result=False)
    def get_large_dataset():
        ...

Enhanced Prompt Tracing
^^^^^^^^^^^^^^^^^^^^^^^

Mark functions for automatic prompt enhancement analysis::

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

When ``enh_prompt=True``, the decorator generates a unique ``enh_prompt_id`` for each function call.
The JSON output includes:

- ``enh_prompt_id``: Unique identifier for each enhanced prompt trace
- ``auto_enhance_after``: Configured threshold for auto-enhancement
- ``enh_prompt_traces``: List of all ``enh_prompt_id`` values in the export

This allows collecting traces across multiple JSON files and rendering them in a UI for analysis.

Pipeline Example
----------------

Chained tasks with multiple API calls::

    python -m example.pipeline.main "Your prompt here"

This runs a three-step pipeline (research → summarize → critique) and writes a single JSON file with all events.

Output
------

By default, ``observer.flush()`` writes ``./llm_observability.json``. Override with the ``LLM_OBS_OUT`` environment variable::

    LLM_OBS_OUT=/path/to/output.json python your_script.py

Async Flush (Non-Blocking Server Upload)
----------------------------------------

When ``observer.flush()`` is called, trace data is automatically sent to the aiobs cloud server
for ingestion. This upload happens **asynchronously in a background thread**, ensuring your
application is not blocked waiting for the network request to complete.

How It Works
^^^^^^^^^^^^

1. ``flush()`` writes data locally (to file or exporter) immediately
2. A background daemon thread sends the data to the aiobs flush server
3. Your application continues execution without waiting for the upload
4. Errors during upload are logged but do not raise exceptions

Example::

    from aiobs import observer

    observer.observe(api_key="aiobs_sk_...")

    # ... your LLM calls ...

    observer.end()
    observer.flush()  # Returns immediately after local write
    # Background thread handles server upload asynchronously

    # Your app continues without delay
    print("Done!")

Benefits
^^^^^^^^

- **Non-blocking**: Main application thread is never blocked by network I/O
- **Fault-tolerant**: Upload failures are logged but don't crash your application
- **Automatic**: No additional configuration required—async upload is enabled by default

.. note::

   The background upload requires a valid API key. If no API key is configured,
   data is only written locally and not sent to the server.

Error Handling
^^^^^^^^^^^^^^

Upload errors are handled gracefully and logged as warnings:

- **401 Unauthorized**: Invalid API key
- **429 Too Many Requests**: Rate limit exceeded
- **Network errors**: Connection failures

Check your application logs for upload status messages when debugging.

What Gets Captured
------------------

For each session:

- **Session ID**: Unique identifier
- **Session name**: Optional custom name
- **Labels**: Key-value pairs for filtering (user-defined + system labels)
- **Metadata**: Process ID, working directory
- **Timing**: start/end timestamps

For each LLM API call:

- **Provider**: ``openai`` or ``gemini``
- **API**: e.g., ``chat.completions.create``, ``embeddings.create``, ``models.generate_content``, or ``models.generate_videos``
- **Request**: model, messages/contents/input/prompt, core parameters
- **Response**: text (for completions), embeddings (for embeddings API), operation info (for video generation), model, token usage (when available)
- **Timing**: start/end timestamps, ``duration_ms``
- **Errors**: exception name and message if the call fails
- **Callsite**: file path, line number, and function name where the API was called

For decorated functions (``@observe``):

- Function name and module
- Input arguments (args/kwargs)
- Return value
- Timing: start/end timestamps, ``duration_ms``
- Errors: exception name and message if the call fails
- Callsite: file path, line number where the function was defined
- Enhanced prompt metadata (``enh_prompt_id``, ``auto_enhance_after``) when enabled
