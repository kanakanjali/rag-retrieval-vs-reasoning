"""
llm_answerer.py — calls an LLM to answer a question given retrieved context.

Uses the OpenAI-compatible chat completions API. Works with OpenAI directly,
or with any OpenAI-compatible provider (Groq, Together, etc.) by changing
OPENAI_BASE_URL in your .env file — handy if you want a free-tier option.

Requires: pip install openai python-dotenv
Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, OPENAI_MODEL) in a .env
file in the project root — see .env.example.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,  # None = default OpenAI endpoint
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


PROMPT_TEMPLATES = {
    "direct": (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say \"I don't know.\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "step_by_step": (
        "Read the context carefully, then answer the question step by step, "
        "citing which part of the context you used. "
        "If the context doesn't contain the answer, say \"I don't know.\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n"
        "Let's think step by step, then give a final answer on the last line "
        "prefixed with 'Final answer:'."
    ),
    "concise": (
        "Using only the context, answer in one short sentence. "
        "If not answerable from the context, say \"I don't know.\"\n\n"
        "Context:\n{context}\n\nQ: {question}\nA:"
    ),
}


def answer_question(question, retrieved_docs, prompt_style="direct"):
    """
    retrieved_docs: list of (doc_id, text, score) tuples from a retriever.
    prompt_style: one of PROMPT_TEMPLATES keys.
    Returns the raw LLM answer text.
    """
    context = "\n".join(f"[{doc_id}] {text}" for doc_id, text, _ in retrieved_docs)
    prompt = PROMPT_TEMPLATES[prompt_style].format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # quick manual sanity check — requires OPENAI_API_KEY to be set
    fake_docs = [("d006", "git stash temporarily shelves uncommitted changes...", 0.8)]
    print(answer_question("What does git stash do?", fake_docs))
