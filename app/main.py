import asyncio
import json
import os
import re
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.retrieval import HybridRetriever, QdrantHybridRetriever, chunk_text

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "public_sources.json"
QDRANT_DIR = ROOT / "qdrant_db"

qdrant_retriever: QdrantHybridRetriever | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global qdrant_retriever
    if QDRANT_DIR.exists():
        qdrant_retriever = await asyncio.to_thread(QdrantHybridRetriever, QDRANT_DIR)
    elif os.getenv("REFRESH_SOURCES_ON_START", "false").lower() == "true":
        try:
            from scraper.scrape import merge_sources, scrape

            scraped = await asyncio.to_thread(scrape)
            if scraped:
                combined = merge_sources(scraped)
                DATA_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"Source refresh failed; using existing data: {exc}")
    yield
    if qdrant_retriever is not None:
        qdrant_retriever.close()
        qdrant_retriever = None


app = FastAPI(title="DarGlobal Applied AI Chatbot", version="1.0", lifespan=lifespan)
conversations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
MAX_CONVERSATION_SESSIONS = 1000
MAX_MESSAGES = 15


def load_documents() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    raw_documents = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    for document in raw_documents:
        if document.get("chunk_id"):
            chunks.append(document)
            continue
        document_chunks = chunk_text(
            document["text"],
            document["url"],
            document["source"],
            document["title"],
            document["collected_at"],
        )
        chunks.extend(document_chunks)
    return chunks


def retrieve(query: str, limit: int = 5, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    prior_user_text = " ".join(
        message["content"] for message in (history or [])[-6:] if message["role"] == "user"
    )
    query_text = (
        f"{prior_user_text} {query}".strip()
        if prior_user_text and re.search(r"\b(it|there|that|this|them|those)\b", query, re.IGNORECASE)
        else query
    )

    if qdrant_retriever is not None:
        return qdrant_retriever.retrieve(query_text, limit=limit)

    documents = load_documents()
    return HybridRetriever(documents).retrieve(query_text, limit)


def demo_answer(query: str, documents: list[dict[str, Any]]) -> str:
    if not documents:
        return (
            "I could not find an answer in the currently collected public sources. "
            "Please ask about DarGlobal projects, locations, property types, or Wasalt services."
        )
    bullets = [f"- {document['title']}: {document['text']}" for document in documents]
    sources = "\n".join(
        f"{index}. [{document['title']}]({document['url']})"
        for index, document in enumerate(documents, start=1)
    )
    return (
        "Based on the collected public data:\n\n"
        + "\n".join(bullets)
        + "\n\nSources:\n"
        + sources
    )


async def openrouter_answer(
    query: str,
    documents: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> str:
    if not documents:
        return demo_answer(query, [])
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return demo_answer(query, documents)

    context = "\n\n".join(
        f"Title: {document['title']}\n"
        f"URL: {document['url']}\n"
        f"Type: {document.get('source_type', 'general_page')} | "
        f"Policy Status: {document.get('policy_status', 'active')} | "
        f"Country: {document.get('countries', 'Unspecified')}\n"
        f"Content: {document['text']}"
        for document in documents
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI real estate assistant for DarGlobal and Wasalt. "
                "Answer ONLY from the provided public sources and recent conversation, "
                "and respond in the same language as the user's question (English or Arabic). "
                "Rules:\n"
                "1. Source Priority: Trust 'project_page', 'platform_listing', and 'general_page' over 'blog_or_press' if facts conflict.\n"
                "2. Legacy Policy Guardrail: If any source has Policy Status 'legacy_verify' (such as Spain's former real estate Golden Visa, which closed in 2025), "
                "explicitly warn the user that the residency or tax program mentioned in that article may no longer be active.\n"
                "3. Citations & Accuracy: Cite source URLs for your claims. If the sources do not support an answer, say so clearly. "
                "Never invent prices, availability, unit counts, handover dates, or legal terms."
            ),
        },
        *history,
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
    ]

    preferred = os.getenv("OPENROUTER_MODEL", "").strip()
    candidate_models = [
        m for m in [preferred, "openrouter/free", "google/gemma-3-27b-it:free", "meta-llama/llama-3.3-70b-instruct:free"] if m
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": "DarGlobal Applied AI Chatbot",
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        for model_name in candidate_models:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content.strip()
            except (httpx.HTTPError, KeyError, IndexError):
                continue

    return demo_answer(query, documents)


@app.get("/health")
async def health() -> dict[str, str]:
    doc_count = qdrant_retriever.count() if qdrant_retriever is not None else len(load_documents())
    return {
        "status": "healthy",
        "documents": str(doc_count),
        "retriever": "qdrant_hybrid" if qdrant_retriever is not None else "tfidf_bm25",
    }


@app.get("/api/sources")
async def sources() -> dict[str, list[dict[str, Any]]]:
    return {"sources": load_documents()[:200]}


@app.get("/api/source-manifest")
async def source_manifest() -> dict[str, Any]:
    manifest_file = ROOT / "data" / "source_manifest.json"
    if not manifest_file.exists():
        return {"manifest": []}
    return {"manifest": json.loads(manifest_file.read_text(encoding="utf-8"))}


@app.post("/api/chat")
async def chat(payload: dict[str, str]) -> dict[str, str]:
    query = payload.get("message", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message is required")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Message is too long")

    session_id = payload.get("session_id", "").strip() or str(uuid.uuid4())
    history = conversations.pop(session_id, [])
    documents = await asyncio.to_thread(retrieve, query, 5, history)
    answer = await openrouter_answer(query, documents, history)

    history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])
    history = history[-MAX_MESSAGES:]
    conversations[session_id] = history
    if len(conversations) > MAX_CONVERSATION_SESSIONS:
        conversations.popitem(last=False)

    return {"answer": answer, "session_id": session_id}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "app" / "index.html").read_text(encoding="utf-8"))