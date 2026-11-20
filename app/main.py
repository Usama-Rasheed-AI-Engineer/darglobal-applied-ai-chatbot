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
ENRICHED_FILE = ROOT / "data" / "enriched_rag_data.jsonl"
QDRANT_DIR = ROOT / "qdrant_db"

qdrant_retriever: QdrantHybridRetriever | None = None
lexical_retriever: HybridRetriever | None = None


def load_documents() -> list[dict[str, Any]]:
    if ENRICHED_FILE.exists():
        docs: list[dict[str, Any]] = []
        with ENRICHED_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = rec.get("metadata", {})
                url = meta.get("source_url") or rec.get("source_url") or rec.get("url", "")
                title = meta.get("title") or rec.get("title") or url or "DarGlobal / Wasalt Source"
                platform = meta.get("platform") or rec.get("platform") or rec.get("source", "")
                scraped = meta.get("last_scraped") or rec.get("last_scraped") or rec.get("collected_at", "")
                source_type = meta.get("source_type") or rec.get("source_type", "general_page")
                policy_status = meta.get("policy_status") or rec.get("policy_status", "active")
                countries = meta.get("countries") or rec.get("countries", "Unspecified")

                for piece in chunk_text(
                    rec.get("text", ""),
                    url=url,
                    source=platform,
                    title=title,
                    collected_at=scraped,
                ):
                    piece["source_type"] = source_type
                    piece["policy_status"] = policy_status
                    piece["countries"] = countries
                    docs.append(piece)
        if docs:
            return docs

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


@asynccontextmanager
async def lifespan(application: FastAPI):
    global qdrant_retriever, lexical_retriever
    use_dense = os.getenv("ENABLE_DENSE_QUERY", "true").lower() == "true"

    if use_dense and QDRANT_DIR.exists():
        qdrant_retriever = await asyncio.to_thread(QdrantHybridRetriever, QDRANT_DIR)
    else:
        if os.getenv("REFRESH_SOURCES_ON_START", "false").lower() == "true":
            try:
                from scraper.scrape import merge_sources, scrape

                scraped = await asyncio.to_thread(scrape)
                if scraped:
                    combined = merge_sources(scraped)
                    DATA_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                print(f"Source refresh failed; using existing data: {exc}")
        docs = await asyncio.to_thread(load_documents)
        lexical_retriever = await asyncio.to_thread(HybridRetriever, docs)

    yield

    if qdrant_retriever is not None:
        qdrant_retriever.close()
        qdrant_retriever = None


app = FastAPI(title="DarGlobal Applied AI Chatbot", version="1.0", lifespan=lifespan)
conversations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
MAX_CONVERSATION_SESSIONS = 1000
MAX_MESSAGES = 15


def retrieve(query: str, limit: int = 6, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
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
    if lexical_retriever is not None:
        return lexical_retriever.retrieve(query_text, limit=limit)

    return HybridRetriever(load_documents()).retrieve(query_text, limit)


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
                "You are a professional real estate assistant for DarGlobal (darglobal.co.uk) and Wasalt (wasalt.sa). "
                "Answer ONLY using facts explicitly stated in the provided Sources below.\n\n"
                "RESPONSE RULES:\n"
                "1. Language Purity: Reply in the exact language of the user's question. When replying in Arabic, write 100% in natural Modern Standard Arabic—never mix English words (such as 'sources', 'availability', 'active', or 'monthlies') into Arabic text.\n"
                "2. No Meta-Talk: Speak naturally as a property advisor. Never mention internal system rules, accuracy policies, or metadata tags like 'general_page', 'platform_listing', or 'Policy Status' in your answer.\n"
                "3. Strict Grounding: Do not guess, extrapolate, or combine facts across different projects. If the Sources do not state an answer (including unstated prices, mortgage rates, rental vs. sale availability, or unlisted cities), clearly state that the collected public sources do not contain that information.\n"
                "4. Source Priority: Prefer project/listing pages over blog or press articles if details conflict.\n"
                "5. Targeted Policy Warning: ONLY if the user's question involves residency, visas, or tax benefits AND a relevant cited source is marked 'legacy_verify' (such as Spain's former Golden Visa), add a brief note advising the user to verify current legal eligibility with a qualified adviser.\n"
                "6. Citations: Include clean Markdown links using only exact URLs from the Sources below."
            ),
        },
        *history,
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
    ]

    preferred = os.getenv("OPENROUTER_MODEL", "").strip()
    candidate_models = list(
        dict.fromkeys(
            m
            for m in [
                preferred,
                "openrouter/free",
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-3.5-lightning:free",
            ]
            if m
        )
    )

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
    if qdrant_retriever is not None:
        doc_count = qdrant_retriever.count()
        mode = "qdrant_dense_sparse"
    elif lexical_retriever is not None:
        doc_count = len(lexical_retriever.documents)
        mode = "enriched_bm25_tfidf"
    else:
        doc_count = len(load_documents())
        mode = "fallback"
    return {"status": "healthy", "documents": str(doc_count), "retriever": mode}


@app.get("/api/sources")
async def sources() -> dict[str, list[dict[str, Any]]]:
    if lexical_retriever is not None:
        return {"sources": lexical_retriever.documents[:200]}
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
    documents = await asyncio.to_thread(retrieve, query, 6, history)
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