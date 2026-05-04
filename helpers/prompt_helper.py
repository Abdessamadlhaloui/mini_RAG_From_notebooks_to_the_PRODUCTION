"""
Prompt construction helper (kept for extensibility).
The primary prompt logic lives in GenerationService using ChatPromptTemplate,
but this module provides utility functions for custom prompt formatting.
"""


def format_prompt(template: str, **kwargs) -> str:
    """Safely formats a prompt template with the provided keyword arguments."""
    return template.format(**kwargs)
