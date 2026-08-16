from pydantic import BaseModel
from anthropic import Anthropic

from qdrant_setup import search, get_client
from golden_queries import load_queries

GEN_MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


class GeneratedAnswer(BaseModel):
    answer_text: str
    used_context: bool


ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Answer a question using only the provided context, or explicitly decline.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_text": {
                "type": "string",
                "description": "The answer, grounded only in the given context. If context is insufficient, a clear statement that the answer isn't available in the provided material.",
            },
            "used_context": {
                "type": "boolean",
                "description": "True if the context was sufficient to answer; false if you had to decline.",
            },
        },
        "required": ["answer_text", "used_context"],
    },
}


def generate_answer(query: str, retrieved_texts: list[str]) -> GeneratedAnswer:
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = (
        "Answer the question using ONLY the information in the context below. "
        "If the context does not contain enough information to answer, say so "
        "explicitly rather than guessing or using outside knowledge.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=400,
        temperature=0,
        tools=[ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": prompt}],
    )
    return GeneratedAnswer.model_validate(response.content[0].input)


if __name__ == "__main__":
    queries = load_queries()  # from Lesson A.4
    unanswerable = next(q for q in queries if q.category == "unanswerable")

    qdrant_client = get_client()
    results = search(qdrant_client, unanswerable.query_text, k=5)
    texts = [p.payload["text"] for p in results.points]

    answer = generate_answer(unanswerable.query_text, texts)
    print(f"used_context={answer.used_context}")
    print(answer.answer_text)
