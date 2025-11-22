import os
import sys

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project = "LLM Observability"
author = "LLM Observability Authors"
version = "0.1.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Pick theme dynamically: prefer Furo, fallback to Alabaster if missing
try:  # pragma: no cover - simple import guard
    import furo  # noqa: F401
    HAS_FURO = True
except Exception:  # pragma: no cover
    HAS_FURO = False

html_theme = "furo" if HAS_FURO else "alabaster"

if HAS_FURO:
    html_theme_options = {
        "light_logo": "logo-light.svg",
        "dark_logo": "logo-dark.svg",
        "sidebar_hide_name": True,
        "navigation_with_keys": True,
        # Link to the repository for "View on GitHub" and edit links
        "source_repository": "https://github.com/neuralis/llm-observability/",
        "source_branch": "main",
        "source_directory": "docs/",
        # Footer GitHub icon
        "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/neuralis/llm-observability",
                "html": """
<svg stroke='currentColor' fill='currentColor' stroke-width='0' viewBox='0 0 16 16' height='1.2em' width='1.2em' xmlns='http://www.w3.org/2000/svg'>
  <path fill-rule='evenodd' d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z'></path>
</svg>
""",
                "class": "",
            }
        ],
    }
else:
    html_theme_options = {}
    # For non-Furo themes, set a single logo
    html_logo = "_static/logo-light.svg"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
