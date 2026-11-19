import json
import os
import re
import uuid
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.retrieval import HybridRetriever, chunk_text

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "public_sources.json"

@asynccontextmanager
async def lifespan(application: FastAPI):
    if os.getenv("REFRESH_SOURCES_ON_START", "false").lower() == "true":
        try:
            from scraper.scrape import scrape, merge_sources
            scraped = await asyncio.to_thread(scrape)
            if scraped:
                combined = merge_sources(scraped)
                DATA_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"Source refresh failed; using existing data: {exc}")
    yield


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
        message["content"] for message in (history or [])[-10:] if message["role"] == "user"
    )
    documents = load_documents()
    query_text = f"{prior_user_text} {query}".strip() if re.search(r"\b(it|there|that|this)\b", query, re.IGNORECASE) else query

    explicit_projects = sorted({
        project
        for project in re.findall(
            r"\b(AIDA|Tierra Viva|Amaya|Trump Tower|W Residences|Venice|Da Vinci|Lucia)\b",
            query,
            re.IGNORECASE,
        )
    }, key=len, reverse=True)
    if explicit_projects:
        project = explicit_projects[0].lower()
        documents = [
            document
            for document in documents
            if project in f"{document.get('title', '')} {document.get('text', '')}".lower()
        ]
    else:
        project_phrase = re.search(r"\b(?:at|about)\s+([A-Z][A-Za-z0-9 &-]{2,})(?:\?|$)", query)
        if project_phrase:
            project = project_phrase.group(1).strip().lower()
            documents = [
                document
                for document in documents
                if project in f"{document.get('title', '')} {document.get('text', '')}".lower()
            ]
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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return demo_answer(query, documents)

    context = "\n\n".join(
        f"Title: {document['title']}\nURL: {document['url']}\nContent: {document['text']}"
        for document in documents
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the provided public sources and recent conversation. "
                "Cite source titles and URLs. If the sources do not support an answer, "
                "say that it is unknown. Never invent prices, availability, unit counts, "
                "handover dates, or legal information."
            ),
        },
        *history,
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
    ]
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
                    "X-Title": "DarGlobal Applied AI Chatbot",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError):
        return demo_answer(query, documents)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "documents": str(len(load_documents()))}


@app.get("/api/sources")
async def sources() -> dict[str, list[dict[str, Any]]]:
    return {"sources": load_documents()}


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
    documents = retrieve(query, history=history)
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
