from pydantic import BaseModel
from anthropic import Anthropic
from golden_queries import load_queries
from qdrant_setup import get_client, search

CRITIC_MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


class RetrievalCritique(BaseModel):
    confident: bool
    reasoning: str
    missing_info: str | None = None


CRITIQUE_TOOL = {
    "name": "submit_critique",
    "description": "Judge whether retrieved passages actually answer a query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "confident": {
                "type": "boolean",
                "description": "True only if the retrieved passages contain enough information to fully answer the query.",
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the judgment.",
            },
            "missing_info": {
                "type": "string",
                "description": "If not confident, what specific information is missing. Omit if confident.",
            },
        },
        "required": ["confident", "reasoning"],
    },
}


def critique_retrieval(query: str, retrieved_texts: list[str]) -> RetrievalCritique:
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = (
        f"QUERY: {query}\n\n"
        f"RETRIEVED PASSAGES:\n{context}\n\n"
        "Can this query be fully answered using ONLY the retrieved passages above? "
        "Judge strictly: if the passages are topically related but missing the specific "
        "fact the query needs, that is NOT confident."
    )
    response = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=300,
        temperature=0,
        tools=[CRITIQUE_TOOL],
        tool_choice={"type": "tool", "name": "submit_critique"},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_use = response.content[0]
    return RetrievalCritique.model_validate(tool_use.input)


if __name__ == "__main__":
    queries = load_queries()  
    direct = next(q for q in queries if q.category == "direct_lookup")
    unanswerable = next(q for q in queries if q.category == "unanswerable")

    qdrant_client = get_client()  

    for label, q in [
        ("direct_lookup (expect confident=True)", direct),
        ("unanswerable (expect confident=False)", unanswerable),
    ]:
        results = search(qdrant_client, q.query_text, k=5)
        texts = [p.payload["text"] for p in results.points]
        critique = critique_retrieval(q.query_text, texts)
        print(f"{label}: confident={critique.confident} -- {critique.reasoning}")
