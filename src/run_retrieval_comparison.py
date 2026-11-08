"""
run_retrieval_comparison.py — Experiment 1: HOLD THE PROMPT FIXED, vary retrieval.

Runs every question in data/questions.csv through two retrievers (TF-IDF vs.
embeddings), using the SAME prompt template for both, and saves every
(question, retrieved docs, answer) to results/retrieval_comparison.csv for
manual labeling.

Usage:
    python src/run_retrieval_comparison.py
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from retriever_tfidf import TfidfRetriever
from retriever_embeddings import EmbeddingRetriever
from llm_answerer import answer_question

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOP_K = 1
FIXED_PROMPT_STYLE = "direct"  # prompt is held constant across this experiment


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    questions = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))
    corpus_path = os.path.join(DATA_DIR, "corpus.csv")

    print("Building TF-IDF retriever...")
    tfidf = TfidfRetriever(corpus_path)
    print("Building embedding retriever (downloads model on first run)...")
    embed = EmbeddingRetriever(corpus_path)

    rows = []
    for _, row in questions.iterrows():
        qid, question, expected, gold_doc = (
            row["question_id"], row["question"], row["expected_answer"], row["gold_doc_id"]
        )
        print(f"  {qid}: {question}")

        for retriever_name, retriever in [("tfidf", tfidf), ("embeddings", embed)]:
            retrieved = retriever.retrieve(question, top_k=TOP_K)
            retrieved_ids = [d[0] for d in retrieved]
            gold_was_retrieved = gold_doc in retrieved_ids

            answer = answer_question(question, retrieved, prompt_style=FIXED_PROMPT_STYLE)

            rows.append({
                "question_id": qid,
                "question": question,
                "expected_answer": expected,
                "gold_doc_id": gold_doc,
                "retriever": retriever_name,
                "retrieved_doc_ids": ",".join(retrieved_ids),
                "gold_doc_retrieved": gold_was_retrieved,
                "llm_answer": answer,
                # fill these in by hand after reading llm_answer vs expected_answer:
                "answer_correct": "",       # y / n
                "error_type": "",           # retrieval / reasoning / none (blank if correct)
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "retrieval_comparison.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")
    print("Next: open this CSV and fill in 'answer_correct' (y/n) and, for wrong")
    print("answers, 'error_type' = 'retrieval' (gold_doc_retrieved was False) or")
    print("'reasoning' (gold_doc_retrieved was True but the answer was still wrong).")


if __name__ == "__main__":
    main()
