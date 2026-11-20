import json
import re
import shutil
from pathlib import Path
from tqdm import tqdm
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

INPUT_FILE = Path("data/enriched_rag_data.jsonl")
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "darglobal_wasalt"
BATCH_SIZE = 16  # Saves to disk every 16 chunks to protect against power cuts

def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "، ", " "]:
                cut = text.rfind(sep, start + max_chars // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks

def create_fresh_collection(client: QdrantClient, dense_dim: int):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE.resolve()}")

    # 1. Load and chunk enriched records
    all_chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record["metadata"]
            for idx, piece in enumerate(chunk_text(record["text"])):
                all_chunks.append((piece, {**meta, "chunk_index": idx}))

    total_chunks = len(all_chunks)
    print(f"Prepared {total_chunks} total chunks from {INPUT_FILE}")

    # 2. Load cached models from disk (no re-download)
    print("Loading cached intfloat/multilingual-e5-large and Qdrant/bm25 models...")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    dense_dim = len(next(dense_model.passage_embed(["dimension check"])))

    # 3. Open Qdrant and resume from saved progress if available
    start_index = 0
    try:
        client = QdrantClient(path=DB_PATH)
        if client.collection_exists(COLLECTION_NAME):
            start_index = client.count(collection_name=COLLECTION_NAME).count
            if start_index >= total_chunks:
                print(f"All {total_chunks} chunks are already indexed! Nothing left to do.")
                return
            print(f"Resuming from saved checkpoint: {start_index}/{total_chunks} chunks already indexed.")
        else:
            create_fresh_collection(client, dense_dim)
    except Exception as e:
        print(f"Storage was interrupted mid-write ({e}). Starting clean with 16-chunk checkpoints...")
        shutil.rmtree(DB_PATH, ignore_errors=True)
        client = QdrantClient(path=DB_PATH)
        create_fresh_collection(client, dense_dim)
        start_index = 0

    # 4. Embed and save in small 16-chunk checkpoints
    point_id = start_index
    for i in tqdm(range(start_index, total_chunks, BATCH_SIZE), desc="Indexing chunks"):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [item[0] for item in batch]

        dense_vecs = list(dense_model.passage_embed(texts, batch_size=BATCH_SIZE))
        sparse_vecs = list(sparse_model.embed(texts, batch_size=BATCH_SIZE))

        points = []
        for text_chunk, meta, d_vec, s_vec in zip(
            texts, [item[1] for item in batch], dense_vecs, sparse_vecs
        ):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": d_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload={"text": text_chunk, **meta},
                )
            )
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Done! Indexed {point_id} hybrid vectors into {DB_PATH}")

if __name__ == "__main__":
    main()