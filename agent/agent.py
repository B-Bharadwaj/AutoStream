from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph

from .intents import classify_intent
from .rag import (
    build_vectorstore,
    extract_facts_from_docs,
    format_retrieved,
    load_kb_documents,
    load_vectorstore,
    retrieve,
    save_vectorstore,
)
from .state import AgentState, Message
from .tools import is_valid_email, mock_lead_capture

SYSTEM_POLICY = """You are AutoStream's production lead-qualification assistant.

Hard rules:
- You MUST answer product/pricing/policy questions ONLY using retrieved KB context (RAG).
- If KB context does not contain the answer, say you don't have that info and offer to connect support.
- Never invent prices, features, or policies.
- Lead capture tool is STRICTLY gated:
  - Only when intent is High-intent lead (ready to sign up)
  - Collect: name -> email -> platform
  - Call tool ONLY once after all three are collected.
"""


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

PLATFORM_KEYWORDS = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "insta": "Instagram",
    "tiktok": "TikTok",
    "twitter": "Twitter/X",
    "x": "Twitter/X",
}

def extract_lead_fields(text: str) -> dict:
    t = text.lower()

    # email
    email = None
    m = EMAIL_RE.search(text)
    if m:
        email = m.group(0)

    # platform
    platform = None
    for k, v in PLATFORM_KEYWORDS.items():
        if re.search(rf"\b{k}\b", t):
            platform = v
            break

    # plan
    plan = None
    if "pro" in t:
        plan = "Pro"
    elif "basic" in t:
        plan = "Basic"

    # name (simple heuristic: "my name is X" / "i am X")
    name = None
    m = re.search(r"(my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z\s]{1,30})", text, re.IGNORECASE)
    if m:
        name = m.group(2).strip()

    return {"name": name, "email": email, "platform": platform, "plan": plan}


def should_use_formatter(user_text: str) -> bool:
    """
    Decide when to use deterministic formatting (no LLM).
    This targets: pricing, plans, policies, and common plan feature comparisons.
    """
    t = user_text.lower()
    keywords = [
        "price",
        "pricing",
        "plan",
        "plans",
        "cost",
        "basic",
        "pro",
        "refund",
        "support",
        "24/7",
        "resolution",
        "4k",
        "720p",
        "captions",
        "videos",
        "unlimited",
    ]
    return any(k in t for k in keywords)


def format_pricing_response(facts: dict, sources: str) -> str:
    basic = facts.get("basic", {})
    pro = facts.get("pro", {})
    refund = facts.get("refund_policy")
    support = facts.get("support_policy")

    lines = []

    # Plans summary
    if basic:
        b_parts = []
        if "price_monthly_usd" in basic:
            b_parts.append(f"${basic['price_monthly_usd']}/month")
        if "videos_per_month" in basic:
            b_parts.append(f"{basic['videos_per_month']} videos/month")
        if "resolution" in basic:
            b_parts.append(f"{basic['resolution']} resolution")
        if b_parts:
            lines.append("**Basic Plan**: " + ", ".join(b_parts) + ".")

    if pro:
        p_parts = []
        if "price_monthly_usd" in pro:
            p_parts.append(f"${pro['price_monthly_usd']}/month")
        if "videos_per_month" in pro:
            p_parts.append(f"{pro['videos_per_month']} videos")
        if "resolution" in pro:
            p_parts.append(f"{pro['resolution']} resolution")
        if pro.get("ai_captions"):
            p_parts.append("AI captions")
        if p_parts:
            lines.append("**Pro Plan**: " + ", ".join(p_parts) + ".")

    # Policies (still from retrieved KB)
    if refund:
        lines.append(f"\n**Refunds**: {refund}")
    if support:
        lines.append(f"\n**Support**: {support}")

    # If nothing extracted, show raw retrieved KB chunks (still grounded)
    if not lines:
        raw = facts.get("raw", [])
        return (
            "Here’s what I found in the AutoStream knowledge base:\n\n"
            + "\n".join([f"- {x.get('title','')}: {x.get('content','')}" for x in raw])
            + f"\n\n(From KB: {sources})"
        )

    return "\n".join(lines) + f"\n\n(From KB: {sources})"


def build_app(kb_path: str = "data/knowledge_base.json"):
    # LLM is optional: used only for non-pricing product questions.
    # If your quota is blocked, the code automatically falls back to deterministic formatting.
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0,
        request_timeout=30,
        max_retries=2,
    )

    # Embeddings for FAISS RAG
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Persist vectorstore so it doesn't rebuild every run
    index_path = "data/faiss_index"
    if Path(index_path).exists():
        vs = load_vectorstore(index_path, embeddings)
    else:
        kb_docs = load_kb_documents(kb_path)
        vs = build_vectorstore(kb_docs, embeddings)
        save_vectorstore(vs, index_path)

    def node_classify(state: AgentState) -> Dict[str, Any]:
        # If your intents.py is rule-only, the llm arg is ignored.
        intent = classify_intent(state.user_input, llm)
        print(f"[INTENT] {intent} | user='{state.user_input}'")
        return {"intent": intent}

    def node_greeting(state: AgentState) -> Dict[str, Any]:
        msg = "👋 Happy to help! Ask me about AutoStream plans, pricing, or say *sign up* to get started."
        return {"history": state.history + [Message(role="assistant", content=msg)]}


    def node_rag_answer(state: AgentState) -> Dict[str, Any]:
        print("[RAG] retrieving...")
        docs = retrieve(vs, state.user_input, k=4)
        context, sources = format_retrieved(docs)
        print("[RAG] retrieved:", sources)

        if not context.strip():
            msg = "I don’t have that information in my knowledge base. If you want, I can connect you to support."
            return {
                "history": state.history + [Message(role="assistant", content=msg)],
                "last_retrieved_context": "",
            }

        # Path C: Deterministic formatting for pricing/plans/policies
        if should_use_formatter(state.user_input):
            facts = extract_facts_from_docs(docs)
            answer = format_pricing_response(facts, sources)
            return {
                "history": state.history + [Message(role="assistant", content=answer)],
                "last_retrieved_context": context,
            }

        # Otherwise: try LLM grounded in retrieved context (nicer phrasing),
        # fallback to deterministic formatting on quota/network failures.
        prompt = (
            SYSTEM_POLICY
            + "\n\n"
            + "Use ONLY the context below to answer. If not in context, say you don't have that info.\n\n"
            + f"Context:\n{context}\n\n"
            + f"User: {state.user_input}\n"
            + "Assistant:"
        )

        print("[RAG] generating answer...")
        try:
            answer = llm.invoke(prompt).content
            msg = f"{answer}\n\n(From KB: {sources})"
        except Exception as e:
            print("[RAG] LLM failed, using fallback:", repr(e))
            facts = extract_facts_from_docs(docs)
            msg = format_pricing_response(facts, sources)

        print("[RAG] done")
        return {
            "history": state.history + [Message(role="assistant", content=msg)],
            "last_retrieved_context": context,
        }

    def node_high_intent_router(state: AgentState) -> Dict[str, Any]:
        """
        Start or continue a gated lead-capture flow.

        Upgrade:
        - Supports users providing platform/email in the same message as plan intent,
        e.g., "I want the pro plan for instagram" (platform is prefilled).
        - Keeps the original flow intact (still collects name -> email -> platform),
        but skips asking for fields already captured.
        """
        import re

        EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
        PLATFORMS = ["youtube", "instagram", "tiktok", "twitter", "x"]

        def extract_quick_lead(text: str) -> dict:
            t = text.lower()

            # Email
            email = None
            m = EMAIL_RE.search(text)
            if m:
                email = m.group(0)

            # Platform
            platform = None
            for p in PLATFORMS:
                if re.search(rf"\b{p}\b", t):
                    platform = "Twitter/X" if p == "x" else p.capitalize()
                    break

            # Plan (optional: only used if you add lead.plan to state.py)
            plan = None
            if "pro" in t:
                plan = "Pro"
            elif "basic" in t:
                plan = "Basic"

            return {"email": email, "platform": platform, "plan": plan}

        if state.lead_capture_done:
            msg = "You’re already set — I’ve captured your details. Anything else you want to know about AutoStream?"
            return {"history": state.history + [Message(role="assistant", content=msg)]}

        user_text = state.user_input.strip()

        # If we are not currently awaiting a field, we may still want to prefill from this message
        if state.awaiting_field is None:
            extracted = extract_quick_lead(user_text)

            # Prefill if present (does NOT break original flow)
            if extracted.get("email") and not state.lead.email:
                state.lead.email = extracted["email"]
            if extracted.get("platform") and not state.lead.platform:
                state.lead.platform = extracted["platform"]

            # OPTIONAL: store chosen plan if you added lead.plan in state.py
            # if extracted.get("plan") and getattr(state.lead, "plan", None) is None:
            #     state.lead.plan = extracted["plan"]

            return {
                "lead": state.lead,
                "awaiting_field": "name",
                "history": state.history
                + [Message(role="assistant", content="Awesome — I can help you get started. What’s your name?")],
            }

        value = user_text  # the user's current input, used as candidate field value

        # Name
        if state.awaiting_field == "name":
            # Also allow prefilling email/platform if user typed them along with name
            extracted = extract_quick_lead(value)
            if extracted.get("email") and not state.lead.email:
                state.lead.email = extracted["email"]
            if extracted.get("platform") and not state.lead.platform:
                state.lead.platform = extracted["platform"]

            if len(value) < 2:
                msg = "Could you share your name (just a couple characters is fine)?"
                return {"history": state.history + [Message(role="assistant", content=msg)]}

            state.lead.name = value

            # If email already captured earlier, skip asking email
            if state.lead.email:
                # If platform already captured too, we can execute tool immediately
                if state.lead.platform and not state.lead_capture_done:
                    mock_lead_capture(state.lead.name, state.lead.email, state.lead.platform)
                    msg = "Perfect — you’re all set ✅ I’ve captured your details and someone from AutoStream will reach out shortly."
                    return {
                        "lead": state.lead,
                        "lead_capture_done": True,
                        "awaiting_field": None,
                        "history": state.history + [Message(role="assistant", content=msg)],
                    }

                return {
                    "lead": state.lead,
                    "awaiting_field": "platform",
                    "history": state.history
                    + [Message(role="assistant", content="Nice to meet you! Which creator platform are you on (YouTube, Instagram, TikTok, etc.)?")],
                }

            return {
                "lead": state.lead,
                "awaiting_field": "email",
                "history": state.history + [Message(role="assistant", content="Nice to meet you! What’s your email?")],
            }

        # Email
        if state.awaiting_field == "email":
            extracted = extract_quick_lead(value)
            candidate_email = extracted.get("email") or value

            # Also allow capturing platform early if present
            if extracted.get("platform") and not state.lead.platform:
                state.lead.platform = extracted["platform"]

            if not is_valid_email(candidate_email):
                msg = "That email doesn’t look valid. Can you re-check and send it again?"
                return {"history": state.history + [Message(role="assistant", content=msg)]}

            state.lead.email = candidate_email

            # If platform already captured earlier, execute tool now
            if state.lead.platform and state.lead.name and not state.lead_capture_done:
                mock_lead_capture(state.lead.name, state.lead.email, state.lead.platform)
                msg = "Perfect — you’re all set ✅ I’ve captured your details and someone from AutoStream will reach out shortly."
                return {
                    "lead": state.lead,
                    "lead_capture_done": True,
                    "awaiting_field": None,
                    "history": state.history + [Message(role="assistant", content=msg)],
                }

            return {
                "lead": state.lead,
                "awaiting_field": "platform",
                "history": state.history
                + [Message(role="assistant", content="Got it. Which creator platform are you on (YouTube, Instagram, TikTok, etc.)?")],
            }

        # Platform
        if state.awaiting_field == "platform":
            extracted = extract_quick_lead(value)
            platform_value = extracted.get("platform") or value

            if len(platform_value) < 2:
                msg = "Which platform do you mainly create on? (e.g., YouTube / Instagram / TikTok)"
                return {"history": state.history + [Message(role="assistant", content=msg)]}

            state.lead.platform = platform_value

            # Strict gating: tool executes only if all fields exist and tool not yet executed.
            if (state.lead.name and state.lead.email and state.lead.platform) and not state.lead_capture_done:
                mock_lead_capture(state.lead.name, state.lead.email, state.lead.platform)
                msg = "Perfect — you’re all set ✅ I’ve captured your details and someone from AutoStream will reach out shortly."
                return {
                    "lead": state.lead,
                    "lead_capture_done": True,
                    "awaiting_field": None,
                    "history": state.history + [Message(role="assistant", content=msg)],
                }

            msg = "I’m missing one detail to proceed. What’s your name, email, and platform?"
            return {"history": state.history + [Message(role="assistant", content=msg)]}

        return {"history": state.history + [Message(role="assistant", content="Let’s continue — what’s your name?")]}

    def route(state: AgentState) -> str:
        t = state.user_input.lower().strip()

        exit_tokens = {
            "thanks", "thank you", "thx",
            "bye", "goodbye", "see you",
            "exit", "quit",
        }

        # Always end the flow on exit phrases (even mid lead-capture)
        if t in exit_tokens:
            return "farewell"

        # If mid lead-capture, continue collecting fields.
        # Only detour to RAG if it's clearly a product question.
        if state.awaiting_field is not None and not state.lead_capture_done:
            resume_tokens = {"ok", "okay", "cool", "next", "continue", "yes", "yep", "sure", "go on", "proceed"}

            # if it's just "ok/next", continue lead flow
            if t in resume_tokens:
                return "high_intent"

            # if user asks a product question mid-flow, answer it via RAG
            if ("?" in t) or should_use_formatter(t):
                return "rag"

            # otherwise treat as field input (name/email/platform)
            return "high_intent"

        # Normal routing by intent
        if state.intent == "Casual greeting":
            return "greeting"
        if state.intent == "Product / pricing inquiry":
            return "rag"
        if state.intent == "High-intent lead (ready to sign up)":
            return "high_intent"

        return "rag"


    def node_farewell(state: AgentState) -> Dict[str, Any]:
        msg = "Thanks! 👋 If you need anything else about AutoStream, feel free to reach out anytime."
        return {"history": state.history + [Message(role="assistant", content=msg)]}


    graph = StateGraph(AgentState)
    graph.add_node("classify", node_classify)
    graph.add_node("greeting", node_greeting)
    graph.add_node("rag", node_rag_answer)
    graph.add_node("high_intent", node_high_intent_router)
    graph.add_node("farewell", node_farewell)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {
            "greeting": "greeting",
            "rag": "rag",
            "high_intent": "high_intent",
            "farewell": "farewell",
        }
    )

    graph.add_edge("farewell", END)
    graph.add_edge("greeting", END)
    graph.add_edge("rag", END)
    graph.add_edge("high_intent", END)

    app = graph.compile()
    return app
