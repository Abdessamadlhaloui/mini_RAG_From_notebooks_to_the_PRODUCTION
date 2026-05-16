from typing import List, Optional
from config.settings import get_settings
from models.message_model import MessageModel
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)
def trim_history_to_budget(history: List[MessageModel], max_tokens: Optional[int]=None) -> List[MessageModel]:
    settings = get_settings()
    budget = max_tokens or settings.MAX_HISTORY_TOKENS
    if not history:
        return []
    kept: List[MessageModel] = []
    used_tokens = 0
    for msg in reversed(history):
        msg_tokens = estimate_tokens(msg.content)
        if used_tokens + msg_tokens > budget and len(kept) >= 2:
            break
        kept.append(msg)
        used_tokens += msg_tokens
    return list(reversed(kept))
def format_history_for_prompt(history: List[MessageModel]) -> List[dict]:
    return [{'role': msg.role, 'content': msg.content} for msg in history]