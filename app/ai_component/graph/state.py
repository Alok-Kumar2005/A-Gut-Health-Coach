from typing import TypedDict, List

class AICompanionState(TypedDict):
    messages: List[str]
    route: str