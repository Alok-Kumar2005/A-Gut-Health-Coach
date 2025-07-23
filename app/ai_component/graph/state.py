from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AICompanionState(TypedDict):
    messages: List[BaseMessage]
    route: str
    conversation_history: str
    user_context: Dict[str, Any]
    session_id: str