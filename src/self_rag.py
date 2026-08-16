from anthropic import Anthropic
from pydantic import BaseModel
from critic import critique_retrieval
from qdrant_setup import search, get_client

CRITIC_MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


class SelfRAGResult(BaseModel):
    final_chunk_ids: list[str]
    attempts: int
    confident: bool
    queries_tried: list[str]


REFORMULATE_TOOL = {
    "name": "submit_reformulation",
    "description": "Rewrite a search query to surface information missing from a previous retrieval attempt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reformulated_query": {
                "type": "string",
                "description": "A differently-worded search query aimed at finding the missing information.",
            }
        },
        "required": ["reformulated_query"],
    },
}


def reformulate_query(original_query: str, missing_info: str) -> str:
    prompt = (
        f"ORIGINAL QUERY: {original_query}\n"
        f"WHAT WAS MISSING FROM THE LAST RETRIEVAL: {missing_info}\n\n"
        "Write a differently-worded search query more likely to surface the missing information. "
        "Do not just repeat the original query."
    )
    response = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=200,
        temperature=0,
        tools=[REFORMULATE_TOOL],
        tool_choice={"type": "tool", "name": "submit_reformulation"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].input["reformulated_query"]


def self_aware_retrieve(client, original_query: str, k: int = 5, max_retries: int = 2) -> SelfRAGResult:
    query = original_query
    queries_tried: list[str] = []

    for attempt in range(1, max_retries + 2):  # 1 initial attempt + max_retries retries
        queries_tried.append(query)
        results = search(client, query, k=k)
        chunk_ids = [p.payload["chunk_id"] for p in results.points]
        texts = [p.payload["text"] for p in results.points]

        critique = critique_retrieval(query, texts)
        is_last_attempt = attempt == max_retries + 1

        if critique.confident or is_last_attempt:
            return SelfRAGResult(
                final_chunk_ids=chunk_ids,
                attempts=attempt,
                confident=critique.confident,
                queries_tried=queries_tried,
            )

        query = reformulate_query(original_query, critique.missing_info or "unspecified")

    raise RuntimeError("unreachable")  



    