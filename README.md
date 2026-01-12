# AutoStream Social-to-Lead Agent (Agentic RAG + LangGraph)

Production-style Conversational AI Agent for a fictional SaaS product **AutoStream** that converts social media conversations into **qualified leads** using:

- **Intent detection** (3-class classification)
- **RAG** (Retrieval-Augmented Generation) over a local knowledge base
- **State management** (multi-turn memory) via **LangGraph**
- **Strict gated tool execution** for lead capture (name → email → platform → tool call)

This is a production-style agent workflow, not a prompt-based chatbot.

---

## What this agent can do

1) Answer product questions only using the **local knowledge base** (RAG)  
- Detect user intent into exactly one of:
    1) Casual greeting  
    2) Product / pricing inquiry  
    3) High-intent lead (ready to sign up)  
-Maintain memory across 5–6+ turns (history + lead capture progress)  
-Capture leads with strict gating:
        - Ask **Name**
        - Ask **Email**
        - Ask **Creator platform**
- Then call `mock_lead_capture()` **once**  

2. Handles interruptions mid-flow (e.g., asking refund policy during signup)

---

## Tech Stack

- **Python:** 3.9+
- **LangChain:** embeddings + vector store + retrieval utilities
- **LangGraph:** agent routing + state management across turns
- **Vector DB:** FAISS (local)
- **LLM:** Gemini (Google Generative AI via `langchain_google_genai`)
- **Knowledge Base:** local JSON (`data/knowledge_base.json`)

> Note: Pricing/policy answers are always grounded in retrieved KB context. The system never invents prices/features.

---

## Project Structure
```
project/
├── agent/
│ ├── main.py # CLI runner
│ ├── agent.py # LangGraph orchestration + routing
│ ├── intents.py # Intent classification (rule-based)
│ ├── rag.py # KB loader, chunking, FAISS, retrieval, fact extraction
│ ├── state.py # Pydantic state schema (memory + lead fields)
│ ├── tools.py # mock_lead_capture + email validation
├── data/
│ └── knowledge_base.json
├── requirements.txt
├── README.md

```

---

## Knowledge Base

The agent’s product knowledge is stored locally in `data/knowledge_base.json`.  
Examples of what’s inside:

- **Basic Plan:** $29/month, 10 videos/month, 720p
- **Pro Plan:** $79/month, unlimited videos, 4K, AI captions
- **Policies:** refund window, support policy

**Hard rule:** product/pricing/policy answers must come only from this KB using RAG.

---

## How It Works (High-level)

Every user message goes through this pipeline:

1) **Intent Classification**
   - Classifies into exactly one of the 3 intents
2) **LangGraph Routing**
   - Routes to one of:
     - `greeting` node (casual)
     - `rag` node (product/pricing inquiry)
     - `high_intent` node (lead capture flow)
     - `farewell` node (bye/thanks/exit)
3) **RAG Answering**
   - For product questions, retrieves relevant KB chunks from FAISS
   - Generates answer grounded in KB
4) **Lead Capture Tool**
   - Strictly gated: executes only after name/email/platform are collected

---

## Architecture Explanation 

This project uses **LangGraph** because the agent needs deterministic, production-style control flow rather than a single prompt chain. A conversational lead agent must enforce strict rules (e.g., collect lead details in order, prevent duplicate tool calls, handle interruptions) and maintain multi-turn context reliably. LangGraph provides a clean state machine abstraction where each message is processed through nodes, and routing decisions are made explicitly through conditional edges.  

For product QA, the agent uses **RAG** over a local knowledge base. KB documents are loaded from JSON, chunked with a text splitter, embedded using Google embeddings, and stored in a local **FAISS** index. Each product/pricing question triggers retrieval of the most relevant chunks; the response is then produced only using retrieved context (with source IDs included). This prevents hallucination and ensures answers remain aligned with the product’s approved information.  

State is managed via a typed Pydantic schema (`AgentState`) that persists conversation history, current intent, lead fields (name/email/platform), and gating flags (`awaiting_field`, `lead_capture_done`). This allows memory across 5–6 turns and guarantees the lead capture tool runs only once after all required fields are collected. The system is designed so it can be deployed behind webhooks (WhatsApp/social platforms) by storing `AgentState` per user session (e.g., Redis) and invoking the graph on every incoming message.

---

## State Management (Memory)

### The agent stores:

- `history`: message list (user + assistant)
- `intent`: one of the 3 intents
- `lead`: `{name, email, platform}`
- `awaiting_field`: `"name" | "email" | "platform"` (controls multi-step flow)
- `lead_capture_done`: prevents duplicate tool execution
- `last_retrieved_context`: debug trace for RAG

This is what enables multi-turn behavior like:

- Start signup
- Ask name
- User asks refund policy mid-flow
- Agent answers refund policy
- Resume signup without losing progress

### Smart Lead Capture & State Gating
The agent supports both step-by-step lead collection and single-message extraction (e.g., “I want the Pro plan for Instagram”).  
If email or platform details are mentioned early, they are prefilled without breaking the gated flow.  
A LangGraph state machine strictly controls the order of data collection (name → email → platform).  
The lead capture tool is executed exactly once and only after all required fields are present.

### Optimized LLM Usage
Pricing, features, and policy responses are generated strictly from retrieved knowledge base content (RAG).  
LLM calls are intentionally minimized to intent classification and optional response phrasing for production reliability.

---

## Intent Detection

Intent is classified into exactly one of:

1. **Casual greeting**
   - e.g., “hi”, “hello”, “bye”, “thanks”
2. **Product / pricing inquiry**
   - e.g., “pricing”, “refund policy”, “plans”
3. **High-intent lead**
   - e.g., “I want to sign up”, “let’s go with pro plan”, “I’m interested”

This project uses a rule-based classifier to ensure **predictable behavior** and avoid failures from API quota limitations.

---

## RAG Pipeline

### Step-by-step
1) Load KB from JSON → `Document` objects
2) Chunk documents (RecursiveCharacterTextSplitter)
3) Embed chunks (Google embeddings)
4) Store in FAISS vector DB (persisted to disk)
5) Retrieve top-k chunks for each product question
6) Answer grounded in retrieved context

### Why FAISS?
- Local, fast, no external DB dependency
- Perfect for demos and production prototypes
- Persisted index avoids rebuilding embeddings every run

---

## Strict Tool Gating (Lead Capture)

Lead tool:

```python
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")
```
Rules enforced:

- Tool executes only when intent is high-intent lead
- Must collect fields in order:
    1) name
    2) email
    3) platform

Tool executes only once per session (lead_capture_done=True)

---
## Running Locally
### 1. Create and activate venv

```python
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
### 2.Install dependencies
```python
pip install -r requirements.txt
```
### 3.Set Gemini API key
```python
#Windows PowerShell:
setx GOOGLE_API_KEY "YOUR_KEY"
#macOS/Linux:
export GOOGLE_API_KEY="YOUR_KEY"
```

### 4.Run the agent (CLI)
```python
python -m agent.main
```

## Example Conversation Flow
User: hi  
Agent: greeting  

User: pricing  
Agent: answers via RAG  

User: I want to go with the pro plan  
Agent: detects high-intent  

User: user  
User: user@gmail.com  
User: instagram  

Tool Output:
Lead captured successfully: user, user@gmail.com, instagram

---

## WhatsApp / Social Platform Integration (Webhook Design)

This agent is designed to integrate cleanly with **WhatsApp, Instagram DMs, or web chat** using **webhooks**, without modifying the core agent logic.

The conversational intelligence (intent detection, RAG, state management, and strict tool gating) lives entirely inside the agent. Messaging platforms act as a thin input/output layer.

---
### How Integration Works

1. **Incoming Message**  
   A messaging platform (e.g., WhatsApp Cloud API or Twilio) sends a webhook request containing:
   - `sender_id` (phone number or user ID)
   - `text` (user message)

2. **Session State Loading**  
    The backend loads a per-user `AgentState` from storage (Redis or database) using `sender_id`.  
   This enables true multi-turn memory across messages.

3. **Agent Execution**  
   The message is passed into the LangGraph agent:
   ```python
   out = app.invoke(state)
    ```
    The agent:
    - Classifies intent
    - Routes via LangGraph
    - Uses RAG for product and pricing questions
    - Executes lead capture tools only when strictly allowed

4. **State Persistence & Reply**  
    The updated state is saved back to storage, and the agent’s reply is sent back to the user via the messaging provider’s API.

5. **Response delivery**    
    The assistant’s final message is extracted from:
    ```python
    state.history[-1].content
    ```
    This message is sent back to WhatsApp using the provider’s API.
---
### Why This Design Works Well

1. True multi-turn memory
Each user has an isolated, persistent AgentState, enabling long conversations without confusion.

2. Platform-agnostic agent core
The same agent can serve:
    - WhatsApp
    - Instagram DMs
    - Website chat
    - Slack

    without modification.

3. Production-safe tool execution   
Lead capture tools are:

    - Executed only after required fields are collected
    - Guaranteed to run only once per user
    - Fully state-driven (no prompt guessing)

4. Easy to scale   
Because state is externalized:
    - Multiple webhook servers can run in parallel
    - Any server can handle any user request

5. Clean separation of concerns
- Webhook server → transport
- LangGraph agent → reasoning & control flow
- Vector DB → knowledge retrieval
- Tool layer → side effects

    This mirrors how real SaaS conversational systems are built.