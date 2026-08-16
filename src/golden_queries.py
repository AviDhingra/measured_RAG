from pathlib import Path
from typing import Literal

from embed_chunks import load_chunks 

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
        query_id="q01", query_text="What was the EchoLeak vulnerability in Microsoft 365 Copilot?",
        category="direct_lookup",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::0", "Prompt Injection & Goal Hijacking::1"],
    ),
    RetrievalQuery(
        query_id="q02", query_text="How did attackers get Copilot to leak files without any user having to click anything?",
        category="paraphrase",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::0", "Prompt Injection & Goal Hijacking::1"],
    ),
    RetrievalQuery(
        query_id="q03", query_text="What technique let a backdoor blend into normal enterprise AI traffic without being noticed?",
        category="paraphrase",
        relevant_chunk_ids=["Infrastructure Compromise::0", "Infrastructure Compromise::1"],
    ),
    RetrievalQuery(
        query_id="q04", query_text="Compare how the GitHub MCP incident and the OpenClaw supply-chain incident both abused trust in developer tooling.",
        category="multi_hop",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::1", "Supply Chain Attacks::0"],
    ),
    RetrievalQuery(
        query_id="q05", query_text="What CVE number was assigned to the Slack AI data exfiltration incident?",
        category="unanswerable", relevant_chunk_ids=[],
    ),
    RetrievalQuery(
        query_id="q06", query_text="How were AI agents tricked using hidden instructions?",
        category="ambiguous",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::0", "MCP Attack Vectors::0"],
    ),

    # -- new: direct_lookup --
    RetrievalQuery(
        query_id="q07", query_text="What is the Perplexity Comet Browser Injection incident?",
        category="direct_lookup", relevant_chunk_ids=["Prompt Injection & Goal Hijacking::2"],
    ),
    RetrievalQuery(
        query_id="q08", query_text="What is the WhatsApp MCP Chat History Exfiltration proof-of-concept?",
        category="direct_lookup", relevant_chunk_ids=["MCP Incidents & PoCs::0"],
    ),
    RetrievalQuery(
        query_id="q09", query_text="What is Tool Shadowing, as an MCP attack vector?",
        category="direct_lookup", relevant_chunk_ids=["MCP Attack Vectors::0"],
    ),

    # -- new: paraphrase --
    RetrievalQuery(
        query_id="q10", query_text="How did a support-ticket processing system end up leaking sensitive credentials because of attacker-supplied SQL?",
        category="paraphrase", relevant_chunk_ids=["Prompt Injection & Goal Hijacking::2"],
    ),
    RetrievalQuery(
        query_id="q11", query_text="How did attackers exploit an AI coding tool through a config file to compromise a developer's machine?",
        category="paraphrase", relevant_chunk_ids=["Supply Chain Attacks::2"],
    ),

    # -- new: multi_hop --
    RetrievalQuery(
        query_id="q12", query_text="Is the GitHub MCP Prompt Injection incident a real-world example of the 'Tool Poisoning' attack vector, or a different mechanism?",
        category="multi_hop",
        relevant_chunk_ids=["MCP Incidents & PoCs::0", "MCP Attack Vectors::0"],
    ),
    RetrievalQuery(
        query_id="q13", query_text="Compare the Financial Reconciliation Agent Fraud incident with the ServiceNow Now Assist Inter-Agent Spoofing incident -- how did each exploit trust in a legitimate-looking business process?",
        category="multi_hop",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::3", "Agent Misalignment & Rogue Behavior::0"],
    ),
    RetrievalQuery(
        query_id="q14", query_text="Compare Procurement Agent Memory Poisoning with the 'Rug Pull / Silent Redefinition' MCP attack vector -- do both describe trust being silently violated after it was first established?",
        category="multi_hop",
        relevant_chunk_ids=["Prompt Injection & Goal Hijacking::4", "MCP Attack Vectors::0"],
    ),

    # -- new: unanswerable --
    RetrievalQuery(
        query_id="q15", query_text="What patch version fixed the Ray Framework CVE-2023-48022 vulnerability?",
        category="unanswerable", relevant_chunk_ids=[],
    ),
    RetrievalQuery(
        query_id="q16", query_text="How many dollars in damages resulted from the OpenClaw/ClawHub malicious skills incident?",
        category="unanswerable", relevant_chunk_ids=[],
    ),
    RetrievalQuery(
        query_id="q17", query_text="What specific exploit technique did the OpenAI evaluation model use to escape its test sandbox in the Hugging Face breach?",
        category="unanswerable", relevant_chunk_ids=[],
    ),

    # -- new: ambiguous --
    RetrievalQuery(
        query_id="q18", query_text="How were AI agent supply chains compromised?",
        category="ambiguous",
        relevant_chunk_ids=["Supply Chain Attacks::0", "Supply Chain Attacks::1"],
    ),
    RetrievalQuery(
        query_id="q19", query_text="What's an example of an AI agent being compromised through a legitimate, trusted communication channel?",
        category="ambiguous",
        relevant_chunk_ids=["Infrastructure Compromise::1", "Prompt Injection & Goal Hijacking::4"],
    ),
    RetrievalQuery(
        query_id="q20", query_text="What's an example of an AI security incident where no external attacker was involved at all?",
        category="ambiguous",
        relevant_chunk_ids=["Agent Misalignment & Rogue Behavior::0", "Agent Misalignment & Rogue Behavior::1"],
    ),
]

def validate_golden_set(queries: list["RetrievalQuery"]) -> None:
    chunks = load_chunks()  # from Lesson A.1
    real_ids = {c.chunk_id for c in chunks}

    broken = 0
    for q in queries:
        for cid in q.relevant_chunk_ids:
            if cid not in real_ids:
                print(f"MISSING: query {q.query_id!r} references {cid!r}, not found in chunks.jsonl")
                broken += 1

    if broken == 0:
        print(f"All relevant_chunk_ids across {len(queries)} queries exist in the current corpus.")
    else:
        print(f"{broken} broken reference(s) -- fix these before running Lesson A.5's evaluation.")


    


if __name__ == "__main__":
    save_queries(examples)
    print(f"Saved {len(examples)} golden queries to {GOLDEN_PATH}")
    validate_golden_set(load_queries())

