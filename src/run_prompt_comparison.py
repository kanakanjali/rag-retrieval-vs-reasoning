"""
run_prompt_comparison.py — Experiment 2: HOLD RETRIEVAL FIXED, vary the prompt.

Pick whichever retriever did better in Experiment 1 (edit FIXED_RETRIEVER
below once you know), then run every question through 3 different prompt
templates, saving every answer to results/prompt_comparison.csv for manual
labeling — same process as Experiment 1, just with the variable flipped.

Usage:
    python src/run_prompt_comparison.py
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from retriever_tfidf import TfidfRetriever
from retriever_embeddings import EmbeddingRetriever
from llm_answerer import answer_question, PROMPT_TEMPLATES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOP_K = 3

# Set this to "tfidf" or "embeddings" based on which won Experiment 1
FIXED_RETRIEVER = "embeddings"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    questions = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))
    corpus_path = os.path.join(DATA_DIR, "corpus.csv")

    print(f"Building {FIXED_RETRIEVER} retriever (fixed for this experiment)...")
    retriever = (TfidfRetriever(corpus_path) if FIXED_RETRIEVER == "tfidf"
                 else EmbeddingRetriever(corpus_path))

    rows = []
    for _, row in questions.iterrows():
        qid, question, expected, gold_doc = (
            row["question_id"], row["question"], row["expected_answer"], row["gold_doc_id"]
        )
        print(f"  {qid}: {question}")
        retrieved = retriever.retrieve(question, top_k=TOP_K)
        retrieved_ids = [d[0] for d in retrieved]
        gold_was_retrieved = gold_doc in retrieved_ids

        for prompt_style in PROMPT_TEMPLATES:
            answer = answer_question(question, retrieved, prompt_style=prompt_style)
            rows.append({
                "question_id": qid,
                "question": question,
                "expected_answer": expected,
                "prompt_style": prompt_style,
                "gold_doc_retrieved": gold_was_retrieved,  # same for all prompt styles here
                "llm_answer": answer,
                "answer_correct": "",   # fill by hand: y / n
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "prompt_comparison.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")
    print("Next: open this CSV and fill in 'answer_correct' (y/n) per row.")


if __name__ == "__main__":
    main()
