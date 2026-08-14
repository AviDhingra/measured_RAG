from pathlib import Path
from typing import Literal

from pydantic import BaseModel

GOLDEN_PATH = Path("../corpus/incidents/golden_queries.jsonl")

Category = Literal["direct_lookup", "paraphrase", "multi_hop", "unanswerable", "ambiguous"]


class RetrievalQuery(BaseModel):
    query_id: str
    query_text: str
    category: Category
    relevant_chunk_ids: list[str]


def save_queries(queries: list["RetrievalQuery"]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")


def load_queries() -> list["RetrievalQuery"]:
    queries = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            queries.append(RetrievalQuery.model_validate_json(line))
    return queries


examples = [
    RetrievalQuery(
        query_id="q01",
        query_text="What was the EchoLeak vulnerability in Microsoft 365 Copilot?",
        category="direct_lookup",
        relevant_chunk_ids=["<your chunk_id from Prompt Injection & Goal Hijacking>"],
    ),
    RetrievalQuery(
        query_id="q02",
        query_text="How did attackers get Copilot to leak files without any user having to click anything?",
        category="paraphrase",  # paraphrases EchoLeak's "zero-click" nature
        relevant_chunk_ids=["<your chunk_id from Prompt Injection & Goal Hijacking>"],
    ),
    RetrievalQuery(
        query_id="q03",
        query_text="What technique let a backdoor blend into normal enterprise AI traffic without being noticed?",
        category="paraphrase",  # paraphrases SesameOp using the OpenAI Assistants API as its C2 channel
        relevant_chunk_ids=["<your chunk_id from Infrastructure Compromise>"],
    ),
    RetrievalQuery(
        query_id="q04",
        query_text="Compare how the GitHub MCP incident and the OpenClaw supply-chain incident both abused trust in developer tooling.",
        category="multi_hop",  # genuinely needs both the GitHub MCP entry AND the OpenClaw/ClawHub entry
        relevant_chunk_ids=[
            "<your chunk_id from Prompt Injection & Goal Hijacking>",
            "<your chunk_id from Supply Chain Attacks>",
        ],
    ),
    RetrievalQuery(
        query_id="q05",
        query_text="What CVE number was assigned to the Slack AI data exfiltration incident?",
        category="unanswerable",  # the corpus entry for this incident doesn't list a CVE at all
        relevant_chunk_ids=[],
    ),
    RetrievalQuery(
        query_id="q06",
        query_text="How were AI agents tricked using hidden instructions?",
        category="ambiguous",  # could mean prompt injection broadly, or MCP tool-description poisoning specifically
        relevant_chunk_ids=[
            "<your chunk_id from Prompt Injection & Goal Hijacking>",
            "<your chunk_id from MCP Attack Vectors>",
        ],
    ),
]

if __name__ == "__main__":
    save_queries(examples)
    print(f"Saved {len(examples)} golden queries to {GOLDEN_PATH}")

