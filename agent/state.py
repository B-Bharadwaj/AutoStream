from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


Intent = Literal["Casual greeting", "Product / pricing inquiry", "High-intent lead (ready to sign up)"]


class LeadDetails(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None
    plan: Optional[str] = None 


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class AgentState(BaseModel):
    # Conversation memory (kept small, but enough for 5–6 turns)
    history: List[Message] = Field(default_factory=list)

    # Latest user message
    user_input: str = ""

    # Current intent (exactly one of the 3)
    intent: Optional[Intent] = None

    # Lead tracking
    lead: LeadDetails = Field(default_factory=LeadDetails)
    lead_capture_done: bool = False

    # Internal flags for lead collection flow
    awaiting_field: Optional[Literal["name", "email", "platform"]] = None

    # RAG answer cache (for debugging / demo)
    last_retrieved_context: Optional[str] = None
