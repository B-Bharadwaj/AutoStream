from __future__ import annotations

from typing import Literal, Optional

Intent = Literal[
    "Casual greeting",
    "Product / pricing inquiry",
    "High-intent lead (ready to sign up)",
]

HIGH_INTENT_HINTS = (
    "sign up", "signup", "start", "get started", "buy", "purchase", "subscribe",
    "trial", "demo", "create account", "register", "i want it", "i'm in", "need pro",
    "im interested", "i'm interested", "interested", "tell me more", "sounds good",
    "i want to try", "wanna try", "i want to start"
)


PRODUCT_HINTS = (
    "price", "pricing", "plan", "plans", "cost", "features", "basic", "pro",
    "refund", "support", "resolution", "4k", "720p", "captions", "videos", "unlimited"
)

GREET_HINTS = (
    "hi", "hello", "hey", "yo",
    "good morning", "good evening",
    "thanks", "thank you", "thx",
    "bye", "goodbye", "see you", "exit", "quit"
)


def _rule_intent(text: str) -> Optional[Intent]:
    t = text.lower().strip()

    decision_words = ("let's", "lets", "go with", "go for", "choose", "pick", "i'll take", "ill take", "get me", "upgrade")
    if any(w in t for w in decision_words) and ("pro" in t):
        return "High-intent lead (ready to sign up)"

    if any(h in t for h in HIGH_INTENT_HINTS):
        return "High-intent lead (ready to sign up)"
    if any(h in t for h in PRODUCT_HINTS):
        return "Product / pricing inquiry"
    if any(t == h or t.startswith(h + " ") for h in GREET_HINTS):
        return "Casual greeting"
    return None


def classify_intent(user_text: str, model=None) -> Intent:
    """
    Production-safe intent classification (no LLM calls).
    - Uses deterministic rules.
    - Defaults to Product/pricing inquiry to keep agent helpful.
    """
    ruled = _rule_intent(user_text)
    if ruled is not None:
        return ruled

    # Safe default for unknown messages
    return "Product / pricing inquiry"
