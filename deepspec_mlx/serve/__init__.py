"""OpenAI-compatible serving for trained DSpark drafts (generic across archs)."""

from .checkpoint import save_draft, load_draft

__all__ = ["save_draft", "load_draft"]
