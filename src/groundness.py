from typing import Literal
from anthropic import Anthropic
from pydantic import BaseModel
from qdrant_setup import search, get_client
from golden_queries import load_queries
from generate_answer import generate_answer

GEN_MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()

class GroundednessVerdict(BaseModel):
    grounded: Literal["full", "partial", "none"]
    unsupported_claims: str | None = None


GROUNDEDNESS_TOOL = {
    "name": "submit_groundedness",
    "description": "Judge whether an answer's claims are supported by its source context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {
                "type": "string",
                "enum": ["full", "partial", "none"],
                "description": "'full' if every claim is supported by the context, 'partial' if some claims are supported and others aren't, 'none' if the answer isn't supported at all.",
            },
            "unsupported_claims": {
                "type": "string",
                "description": "If partial or none, describe specifically what isn't supported. Omit if full.",
            },
        },
        "required": ["grounded"],
    },
}

def score_groundedness(answer_text: str, retrieved_texts: list[str]) -> GroundednessVerdict:
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER TO CHECK:\n{answer_text}\n\n"
        "Is every claim in the answer actually supported by the context above? "
        "Judge strictly: an answer can be topically consistent with the context "
        "while still stating a specific fact the context doesn't actually contain."
    )
    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=300,
        temperature=0,
        tools=[GROUNDEDNESS_TOOL],
        tool_choice={"type": "tool", "name": "submit_groundedness"},
        messages=[{"role": "user", "content": prompt}],
    )
    return GroundednessVerdict.model_validate(response.content[0].input)


if __name__ == "__main__":
    queries = load_queries()
    direct = next(q for q in queries if q.category == "direct_lookup")

    qdrant_client = get_client()
    results = search(qdrant_client, direct.query_text, k=5)
    texts = [p.payload["text"] for p in results.points]

    good_answer = generate_answer(direct.query_text, texts)
    print(f"used_context: {good_answer.used_context}")
    print(f"answer text: {good_answer.answer_text}")
    verdict = score_groundedness(good_answer.answer_text, texts)
    print(f"well-grounded case: {verdict.grounded} -- {verdict.unsupported_claims}")

    # known-bad case: hand-written, not model-generated -- guaranteed
    # reproducible, because it doesn't depend on the model happening to
    # hallucinate on this particular run.
    ray_query = next(q for q in queries if q.query_id == "q15")
    ray_results = search(qdrant_client, ray_query.query_text, k=5)
    ray_texts = [p.payload["text"] for p in ray_results.points]

    fabricated_answer = (
        "The Ray Framework CVE-2023-48022 vulnerability was patched in "
        "version 2.8.1, released in December 2025."
    )
    verdict = score_groundedness(fabricated_answer, ray_texts)
    print(f"fabricated-detail case: {verdict.grounded} -- {verdict.unsupported_claims}")

