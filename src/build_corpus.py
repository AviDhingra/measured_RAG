
import json
from pathlib import Path
from datetime import datetime, timezone
import requests
import re
from pydantic import BaseModel

class IncidentChunk(BaseModel):
    chunk_id: str
    source_section: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    fetched_at: str


RAW_URL = "https://raw.githubusercontent.com/h5i-dev/awesome-ai-agent-incidents/main/README.md"
RAW_DIR = Path("../corpus/incidents/raw")
DOC_PATH = Path("../corpus/incidents/raw/incidents.md")
CHUNKS_PATH = Path("../corpus/incidents/chunks.jsonl")
META_PATH = Path("../corpus/incidents/raw/incidents.meta.json")
CHUNK_SIZE = 800
OVERLAP = 150

WANTED_SECTIONS = [
    "Prompt Injection & Goal Hijacking",
    "Supply Chain Attacks",
    "Infrastructure Compromise",
    "Agent Misalignment & Rogue Behavior",
    "MCP Incidents & PoCs",
    "MCP Attack Vectors",
]

def fetch_corpus() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(RAW_URL, timeout=10)
    resp.raise_for_status()

    doc_path = RAW_DIR / "incidents.md"
    doc_path.write_text(resp.text, encoding="utf-8")

    meta_path = RAW_DIR / "incidents.meta.json"
    meta_path.write_text(json.dumps(
        {"url": RAW_URL, "fetched_at": datetime.now(timezone.utc).isoformat()},
        indent=2,
        ), 
        encoding="utf-8")

    return doc_path


def build_section_map(raw_text: str, wanted: list[str]) -> dict[str, str]:
    headings = list(re.finditer(r"^### (.+?)\s*$", raw_text, re.MULTILINE))

    section_map: dict[str, str] = {}
    for i, m in enumerate(headings):
        name = m.group(1).strip()
        if name not in wanted:
            continue
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw_text)
        section_map[name] = raw_text[start:end].strip()

    missing = set(wanted) - section_map.keys()
    if missing:
        print(f"WARNING: expected sections not found in the corpus: {missing}")

    return section_map

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_corpus(
    section_map: dict[str, str],
    fetched_at: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[IncidentChunk]:
    all_chunks: list[IncidentChunk] = []
    for section_name, text in section_map.items():
        pieces = chunk_text(text, chunk_size, overlap)
        offset = 0
        for i, piece in enumerate(pieces):
            all_chunks.append(
                IncidentChunk(
                    chunk_id=f"{section_name}::{i}",
                    source_section=section_name,
                    chunk_index=i,
                    text=piece,
                    char_start=offset,
                    char_end=offset + len(piece),
                    fetched_at=fetched_at,
                )
            )
            offset += chunk_size - overlap
    return all_chunks

def save_chunks(chunks: list[IncidentChunk]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


if __name__ == "__main__":
    path = fetch_corpus()
    print(f"Fetched corpus to {path}")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    raw_text = DOC_PATH.read_text(encoding="utf-8")
    sections = build_section_map(raw_text, WANTED_SECTIONS)

    chunks = build_corpus(sections, fetched_at=meta["fetched_at"])
    save_chunks(chunks)
    print(f"Wrote {len(chunks)} chunks from {len(sections)} sections to {CHUNKS_PATH}")