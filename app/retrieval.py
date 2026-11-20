from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_text(
    text: str,
    url: str,
    source: str,
    title: str,
    collected_at: str,
    target_chars: int = 700,
    overlap_chars: int = 100,
) -> list[dict[str, str]]:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n{2,}|(?<=[.!?])\s+", text)]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 1 <= target_chars:
            current = f"{current} {paragraph}".strip()
        else:
            if current:
                chunks.append(current)
                current = current[-overlap_chars:].strip()
            while len(paragraph) > target_chars:
                chunk = paragraph[:target_chars].rsplit(" ", 1)[0]
                chunks.append(chunk)
                paragraph = paragraph[len(chunk) - overlap_chars:].lstrip()
            current = f"{current} {paragraph}".strip()
    if current:
        chunks.append(current)

    return [
        {
            "title": title,
            "text": chunk,
            "url": url,
            "source": source,
            "collected_at": collected_at,
            "chunk_id": f"{url}#{index}",
        }
        for index, chunk in enumerate(chunks)
        if len(chunk) >= 40
    ]


class HybridRetriever:
    def __init__(self, documents: Iterable[dict[str, str]]):
        self.documents = list(documents)
        self.corpus = [
            f"{document['title']} {document['text']} {document.get('source', '')}"
            for document in self.documents
        ]
        tokenized = [tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(self.corpus) if self.corpus else None

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        if not self.documents:
            return []

        terms = set(tokenize(query))
        bm25_scores = self.bm25.get_scores(tokenize(query)).tolist() if self.bm25 else [0.0] * len(self.documents)
        if self.tfidf is not None:
            query_vector = self.vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vector, self.tfidf)[0].tolist()
        else:
            tfidf_scores = [0.0] * len(self.documents)

        bm25_order = sorted(range(len(self.documents)), key=lambda i: bm25_scores[i], reverse=True)
        tfidf_order = sorted(range(len(self.documents)), key=lambda i: tfidf_scores[i], reverse=True)

        reciprocal: dict[int, float] = {}
        for rank, index in enumerate(bm25_order[:50], start=1):
            reciprocal[index] = reciprocal.get(index, 0.0) + 1.0 / (60 + rank)
        for rank, index in enumerate(tfidf_order[:50], start=1):
            reciprocal[index] = reciprocal.get(index, 0.0) + 1.0 / (60 + rank)

        reranked: list[tuple[float, int]] = []
        for index, document in enumerate(self.documents):
            text = f"{document['title']} {document['text']}".lower()
            exact_boost = sum(1 for term in terms if term in text)
            title_boost = sum(1 for term in terms if term in document["title"].lower())
            score = reciprocal.get(index, 0.0) + 0.015 * exact_boost + 0.025 * title_boost
            reranked.append((score, index))

        reranked.sort(key=lambda item: item[0], reverse=True)
        return [self.documents[index] for _, index in reranked[:limit]]


class QdrantHybridRetriever:
    """Qdrant retriever supporting full Dense+Sparse RRF locally and low-RAM Sparse mode on 512MB cloud tiers."""

    def __init__(self, db_path: str | Path = "./qdrant_db", collection_name: str = "darglobal_wasalt"):
        from fastembed import SparseTextEmbedding
        from qdrant_client import QdrantClient

        self.db_path = str(db_path)
        self.collection_name = collection_name
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.client = QdrantClient(path=self.db_path)

        self.dense_model = None
        if os.getenv("ENABLE_DENSE_QUERY", "true").lower() == "true":
            from fastembed import TextEmbedding

            self.dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(collection_name=self.collection_name).count

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        from qdrant_client import models

        if not self.client.collection_exists(self.collection_name):
            return []

        sparse_vec = next(self.sparse_model.embed([query]))
        sparse_query = models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )
        dense_vec = next(self.dense_model.query_embed([query])).tolist() if self.dense_model is not None else None

        def _search_bucket(allowed_types: list[str], top_k: int):
            source_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchAny(any=allowed_types),
                    )
                ]
            )
            if dense_vec is not None:
                return self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(query=dense_vec, using="dense", limit=20),
                        models.Prefetch(query=sparse_query, using="sparse", limit=20),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=source_filter,
                    limit=top_k,
                    with_payload=True,
                ).points

            return self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using="sparse",
                query_filter=source_filter,
                limit=top_k,
                with_payload=True,
            ).points

        primary_hits = _search_bucket(["project_page", "platform_listing", "general_page"], max(3, limit - 2))
        secondary_hits = _search_bucket(["blog_or_press"], 2)

        seen: set[tuple[str, Any]] = set()
        results: list[dict[str, Any]] = []
        for hit in list(primary_hits) + list(secondary_hits):
            payload = hit.payload or {}
            url = payload.get("source_url", "")
            dedup_key = (url, payload.get("chunk_index"))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            results.append(
                {
                    "title": payload.get("title") or url or "DarGlobal / Wasalt Source",
                    "text": payload.get("text", ""),
                    "url": url,
                    "source": payload.get("platform", ""),
                    "source_type": payload.get("source_type", "general_page"),
                    "policy_status": payload.get("policy_status", "active"),
                    "countries": payload.get("countries", "Unspecified"),
                    "collected_at": payload.get("last_scraped", ""),
                }
            )
            if len(results) >= limit:
                break

        return results