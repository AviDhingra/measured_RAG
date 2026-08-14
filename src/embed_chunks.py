from pathlib import Path
from ollama import Client
from pydantic import BaseModel
import math

from build_corpus import IncidentChunk

CHUNKS_PATH = Path("../corpus/incidents/chunks.jsonl")
EMBEDDINGS_PATH = Path("../corpus/incidents/embeddings.jsonl")
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
BATCH_SIZE = 16

client = Client(host=OLLAMA_HOST)

class EmbeddedChunk(BaseModel):
    chunk_id: str
    source_section: str
    chunk_index: int
    text: str
    fetched_at: str
    embedding: list[float]

def load_chunks() -> list[IncidentChunk]:
    chunks: list[IncidentChunk] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(IncidentChunk.model_validate_json(line))
    return chunks

def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]

def embed_chunks(chunks: list[IncidentChunk], batch_size: int = BATCH_SIZE) -> list[tuple[IncidentChunk, list[float]]]:
    embedded_chunks: list[tuple[IncidentChunk, list[float]]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_batch(texts)
        embedded_chunks.extend(zip(batch, embeddings))
    return embedded_chunks


def to_embedded_chunks(embedded_chunks: list[tuple[IncidentChunk, list[float]]]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            source_section=c.source_section,
            chunk_index=c.chunk_index,
            text=c.text,
            fetched_at=c.fetched_at,
            embedding=e
        )
        for c, e in embedded_chunks
    ]

def save_embeddings(embedded: list[EmbeddedChunk]) -> None:
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as f:
        for e in embedded:
            f.write(e.model_dump_json() + "\n")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

if __name__ == "__main__":
    chunks =load_chunks()
    embedded_chunks = embed_chunks(chunks)
    embedded = to_embedded_chunks(embedded_chunks)
    save_embeddings(embedded)
    print(f"Saved {len(embedded)} embedded chunks to {EMBEDDINGS_PATH}")

    by_section: dict[str, list[EmbeddedChunk]] = {}
    for e in embedded:
        by_section.setdefault(e.source_section, []).append(e)

    a = by_section["Prompt Injection & Goal Hijacking"][0]
    b = by_section["Infrastructure Compromise"][0]
    print(f"Cosine similarity between '{a.source_section}' and '{b.source_section}': {cosine_similarity(a.embedding, b.embedding)}")