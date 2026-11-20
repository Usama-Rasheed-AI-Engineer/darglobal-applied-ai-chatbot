import sys
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Ensure Windows PowerShell prints Arabic characters without encoding errors
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "darglobal_wasalt"

print("Loading cached query models...")
dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
client = QdrantClient(path=DB_PATH)

def hybrid_search(query: str, primary_k: int = 3, secondary_k: int = 2):
    dense_vec = next(dense_model.query_embed([query])).tolist()
    sparse_vec = next(sparse_model.embed([query]))
    sparse_query = models.SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist(),
    )

    def run_query(allowed_types: list[str], limit: int):
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=20),
                models.Prefetch(query=sparse_query, using="sparse", limit=20),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchAny(any=allowed_types),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        ).points

    # 1. Pull authoritative project/listing pages first
    primary = run_query(["project_page", "platform_listing"], primary_k)
    # 2. Pull blog/press/general pages as secondary context
    secondary = run_query(["blog_or_press", "general_page"], secondary_k)
    return primary + secondary

if __name__ == "__main__":
    test_queries = [
        "Tierra Viva Lamborghini villas in Spain and Golden Visa",
        "شقق للبيع في المدينة المنورة",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        results = hybrid_search(q)
        for rank, hit in enumerate(results, 1):
            p = hit.payload
            snippet = p.get("text", "").replace("\n", " ")[:180]
            print(
                f"{rank}. [score={hit.score:.4f} | type={p.get('source_type')} | "
                f"status={p.get('policy_status')} | country={p.get('countries')}]"
            )
            print(f"   URL: {p.get('source_url')}")
            print(f"   TEXT: {snippet}...\n")