from qdrant_client import QdrantClient, models
import uuid

from embed_chunks import EMBEDDINGS_PATH, client as ollama_client, EMBED_MODEL, EmbeddedChunk

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "incidents"
VECTOR_SIZE = 768
UPSERT_BATCH_SIZE = 32


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

def load_embeddings() -> list[EmbeddedChunk]:
    chunks = []
    with EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(EmbeddedChunk.model_validate_json(line))
    return chunks


def create_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

def upsert_chunks(client, chunks: list[EmbeddedChunk], batch_size: int = UPSERT_BATCH_SIZE) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        points = [
            models.PointStruct(
                id=chunk_id_to_point_id(c.chunk_id),
                vector=c.embedding,
                payload={
                    "chunk_id": c.chunk_id,
                    "source_section": c.source_section,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "fetched_at": c.fetched_at,
                },
            )
            for c in batch
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


def embed_query(text: str) -> list[float]:
    
    response = ollama_client.embed(model=EMBED_MODEL, input=[text])
    return response["embeddings"][0]

def search(client, query_text: str, k: int = 5):
    query_vector = embed_query(query_text)
    return client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=k)


if __name__ == "__main__":
    client = get_client()
    create_collection(client)

    chunks = load_embeddings()
    upsert_chunks(client, chunks)

    count = client.count(collection_name=COLLECTION_NAME)
    print(f"Inserted {count.count} points into Qdrant collection '{COLLECTION_NAME}'")

    results = search(client, "how was OpenAI's API used as a covert command channel?", k=5)
    for point in results.points:
        print(f"Chunk ID: {point.payload['chunk_id']}, Score: {point.score}")
        print(f"Text: {point.payload['text']}\n")