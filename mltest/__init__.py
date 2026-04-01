"""
MLTest - static screening and LLM-assisted unit testing
for system-level programming languages (C, C++, Rust).
"""

__version__ = "0.1.0"
__author__ = "Dissertation Research"

# Load environment variables from .env file when python-dotenv is available.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from .cli import main

__all__ = ["main"]
