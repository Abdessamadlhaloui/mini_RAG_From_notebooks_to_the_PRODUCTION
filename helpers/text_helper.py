import re
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub('\\s+', ' ', text)
    return text
def normalize_text(text: str) -> str:
    return text.lower()
def sanitize_query(query: str) -> str:
    injection_patterns = ['ignore\\s+(all\\s+)?(previous|above|prior)\\s+(instructions|prompts?|rules?)', 'you\\s+are\\s+now\\s+', 'system\\s*:\\s*', '<\\|im_start\\|>', '<\\|im_end\\|>']
    cleaned = query
    for pattern in injection_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)