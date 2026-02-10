"""Compatibility shim for legacy modules importing ``util``."""

try:
    from .myutil import *  # noqa: F401,F403
except ImportError:
    from myutil import *  # noqa: F401,F403
