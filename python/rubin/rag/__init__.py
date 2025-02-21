"""Rubin RAG-basd LLM Application."""

from importlib.metadata import PackageNotFoundError, version

from .chatbot import *
from .layout import *

__all__ = ["__version__"]

__version__: str
"""The version string of rubin.rag
(PEP 440 / SemVer compatible).
"""

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.1"
