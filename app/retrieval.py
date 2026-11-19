from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import json
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
