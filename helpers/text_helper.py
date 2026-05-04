"""
Text sanitization helpers.
Used to clean user-supplied queries before they reach the LLM prompt,
providing a basic defense against prompt injection.
"""
import re


def clean_text(text: str) -> str:
    """Strips leading/trailing whitespace and collapses internal whitespace."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text: str) -> str:
    """Lowercases the text for case-insensitive matching."""
    return text.lower()


def sanitize_query(query: str) -> str:
    """
    Basic prompt-injection sanitization.

    Strips known control phrases that could override system instructions.
    This is NOT a silver bullet — defense-in-depth via strict system prompts
    and ChatPromptTemplate boundaries is the primary mitigation.
    """
    # Remove attempts to override system prompt
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?|rules?)",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
    ]
    cleaned = query
    for pattern in injection_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return clean_text(cleaned)
