from typing import List, Optional
def format_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)
def build_multi_turn_messages(query: str, context_chunks: List[str], history_messages: List[dict], system_prompt: Optional[str]=None) -> List[dict]:
    default_system = 'You are a precise, helpful assistant that answers questions strictly based on the provided document context. If the answer is not in the context, say so clearly. You have access to the conversation history — use it to resolve pronouns and follow-up references.'
    context_text = '\n\n---\n\n'.join(context_chunks) if context_chunks else 'No context retrieved.'
    messages: List[dict] = [{'role': 'system', 'content': system_prompt or default_system}]
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': f'DOCUMENT CONTEXT:\n{context_text}\n\nQUESTION: {query}'})
    return messages