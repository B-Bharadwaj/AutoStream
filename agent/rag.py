from __future__ import annotations

import json
import re
from typing import Dict, Any
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def load_kb_documents(kb_path: str) -> List[Document]:
    data = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    docs = []
    for d in data["documents"]:
        docs.append(
            Document(
                page_content=d["content"],
                metadata={"id": d["id"], "title": d["title"], "product": data.get("product", "AutoStream")},
            )
        )
    return docs


def build_vectorstore(docs: List[Document], embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding=embeddings)


def retrieve(vectorstore: FAISS, query: str, k: int = 4) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)


def format_retrieved(docs: List[Document]) -> Tuple[str, str]:
    """
    Returns:
      - context string
      - compact "sources" string for logging
    """
    context_lines = []
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        doc_id = doc.metadata.get("id", "unknown")
        sources.append(f"{doc_id}:{title}")
        context_lines.append(f"[{title}] {doc.page_content}")
    return "\n".join(context_lines), " | ".join(sources)

def save_vectorstore(vs: FAISS, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(path)

def load_vectorstore(path: str, embeddings) -> FAISS:
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)



MONEY_RE = re.compile(r"\$([0-9]+)")
VIDEOS_RE = re.compile(r"(\d+)\s+videos\/month", re.IGNORECASE)
RES_RE = re.compile(r"(720p|4k)", re.IGNORECASE)

def extract_facts_from_docs(docs: list[Document]) -> Dict[str, Any]:
    """
    Convert retrieved KB docs into structured facts.
    No hardcoding of pricing text—facts are extracted from doc.page_content.
    """
    facts: Dict[str, Any] = {
        "basic": {},
        "pro": {},
        "refund_policy": None,
        "support_policy": None,
        "raw": [],
    }

    for d in docs:
        text = d.page_content.strip()
        facts["raw"].append({"title": d.metadata.get("title", ""), "content": text, "id": d.metadata.get("id", "")})

        t_low = (d.metadata.get("title", "") + " " + text).lower()

        # Basic plan facts
        if "basic plan" in t_low:
            m = MONEY_RE.search(text)
            if m:
                facts["basic"]["price_monthly_usd"] = int(m.group(1))

            mv = VIDEOS_RE.search(text)
            if mv:
                facts["basic"]["videos_per_month"] = int(mv.group(1))

            mr = RES_RE.search(text)
            if mr:
                facts["basic"]["resolution"] = mr.group(1).upper()

        # Pro plan facts
        if "pro plan" in t_low:
            m = MONEY_RE.search(text)
            if m:
                facts["pro"]["price_monthly_usd"] = int(m.group(1))

            if "unlimited videos" in t_low:
                facts["pro"]["videos_per_month"] = "unlimited"

            mr = RES_RE.search(text)
            if mr:
                facts["pro"]["resolution"] = mr.group(1).upper()

            if "ai captions" in t_low:
                facts["pro"]["ai_captions"] = True

        # Policies
        doc_id = (d.metadata.get("id", "") or "").lower()

        if doc_id == "policy_refunds":
            facts["refund_policy"] = text

        if doc_id == "policy_support":
            facts["support_policy"] = text

    return facts
