"""
summarize_results.py — run this AFTER you've manually filled in the
'answer_correct' (and 'error_type') columns in the results CSVs.

Prints:
  - accuracy per retriever (Experiment 1)
  - accuracy per prompt style (Experiment 2)
  - error type breakdown (retrieval-caused vs reasoning-caused) for Experiment 1
  - which moved accuracy more: retrieval choice or prompt choice

Usage:
    python src/summarize_results.py
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _pct_correct(df, group_col):
    g = df.groupby(group_col)["answer_correct"].apply(
        lambda s: (s.str.lower() == "y").mean()
    )
    return (g * 100).round(1)


def main():
    retrieval_path = os.path.join(RESULTS_DIR, "retrieval_comparison.csv")
    prompt_path = os.path.join(RESULTS_DIR, "prompt_comparison.csv")

    if not os.path.exists(retrieval_path) or not os.path.exists(prompt_path):
        print("Run run_retrieval_comparison.py and run_prompt_comparison.py first.")
        return

    r_df = pd.read_csv(retrieval_path)
    p_df = pd.read_csv(prompt_path)

    if r_df["answer_correct"].isna().all() or (r_df["answer_correct"] == "").all():
        print("'answer_correct' is still blank in retrieval_comparison.csv — "
              "label it by hand first (y/n per row), then re-run this script.")
        return

    print("=" * 60)
    print("EXPERIMENT 1 — retrieval strategy comparison (prompt held fixed)")
    print("=" * 60)
    retrieval_acc = _pct_correct(r_df, "retriever")
    print(retrieval_acc.to_string())
    spread_retrieval = retrieval_acc.max() - retrieval_acc.min()
    print(f"\nAccuracy spread across retrievers: {spread_retrieval:.1f} points")

    if "error_type" in r_df.columns and not (r_df["error_type"] == "").all():
        print("\nError type breakdown (wrong answers only):")
        wrong = r_df[r_df["answer_correct"].str.lower() == "n"]
        print(wrong["error_type"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — prompt style comparison (retrieval held fixed)")
    print("=" * 60)
    prompt_acc = _pct_correct(p_df, "prompt_style")
    print(prompt_acc.to_string())
    spread_prompt = prompt_acc.max() - prompt_acc.min()
    print(f"\nAccuracy spread across prompt styles: {spread_prompt:.1f} points")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    bigger = "RETRIEVAL" if spread_retrieval > spread_prompt else "PROMPT"
    print(f"Retrieval-choice spread: {spread_retrieval:.1f} pts | "
          f"Prompt-choice spread: {spread_prompt:.1f} pts")
    print(f"=> {bigger} had the larger effect on answer accuracy in this run.")


if __name__ == "__main__":
    main()
