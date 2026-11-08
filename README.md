# RAG Retrieval vs. Prompt Comparison

A small, self-contained experiment comparing two levers in a
retrieval-augmented question-answering (RAG) pipeline: the **retrieval
method** (keyword-based vs. semantic) and the **prompt template** — to see
which one has a bigger effect on whether the final answer is correct.

## Question this answers

When a RAG pipeline gives a wrong answer, is it usually because:
- the retriever pulled the wrong source documents ("retrieval error"), or
- the retriever found the right documents but the LLM reasoned poorly over
  them ("reasoning error")?

And separately: does changing the retrieval method (TF-IDF vs. sentence
embeddings) move answer accuracy more, or less, than changing the prompt
template?

## Project structure

```
data/
  corpus.csv          25 short reference documents (about git/GitHub concepts)
  questions.csv        25 fixed questions, each with an expected answer and
                        which document actually contains the answer (gold_doc_id)
src/
  retriever_tfidf.py        keyword/n-gram based retrieval
  retriever_embeddings.py   dense semantic retrieval (sentence-transformers)
  llm_answerer.py            calls the LLM with retrieved context + a prompt template
  run_retrieval_comparison.py   Experiment 1: prompt fixed, retriever varied
  run_prompt_comparison.py      Experiment 2: retriever fixed, prompt varied
  summarize_results.py          computes accuracy + error breakdown after labeling
results/              output CSVs land here (created when you run the scripts)
```

## Setup

```
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your API key
```

Swap `data/corpus.csv` and `data/questions.csv` for your own content if you
want to run this on a different domain — the code doesn't assume anything
about the topic.

## Running it

```
python src/run_retrieval_comparison.py   # Experiment 1
python src/run_prompt_comparison.py      # Experiment 2 (edit FIXED_RETRIEVER first,
                                          # based on which won Experiment 1)
```

Both scripts write a CSV to `results/` with every question, the retrieved
docs, and the LLM's answer — but leave the `answer_correct` column blank.
**Open each CSV and fill that column in by hand** (y/n, comparing `llm_answer`
to `expected_answer`), and for Experiment 1, also fill in `error_type`
(`retrieval` if `gold_doc_retrieved` was False, `reasoning` if it was True but
the answer was still wrong).

Then run:

```
python src/summarize_results.py
```

This prints accuracy per retriever, accuracy per prompt style, the error-type
breakdown, and a direct comparison of which lever (retrieval vs. prompt)
moved accuracy more in your run.

## Notes / limitations

- The corpus (25 docs) and question set (25 questions) are small by design,
  so every answer can be manually checked — this trades statistical power
  for the ability to do real error analysis by hand, which matters more at
  this scale than a larger but unchecked dataset.
- `temperature=0` is used for the LLM calls to keep answers as reproducible
  as possible between runs.
- Manual labeling introduces some subjectivity; a single labeler (no
  inter-rater agreement check) is a known limitation of this small setup.
