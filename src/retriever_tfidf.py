"""
retriever_tfidf.py — sparse lexical retrieval using TF-IDF + cosine similarity.

This retrieves purely on word/n-gram overlap. It's fast, needs no model
download, but misses semantic matches that use different wording than the
corpus (e.g. query "undo a bad commit" vs doc text "revert... undoes...").
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus_csv_path, text_col="text", id_col="doc_id"):
        self.df = pd.read_csv(corpus_csv_path)
        self.ids = self.df[id_col].tolist()
        self.texts = self.df[text_col].tolist()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3):
        """Return list of (doc_id, text, score) for the top_k most similar docs."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in ranked_idx]


if __name__ == "__main__":
    # quick manual sanity check
    r = TfidfRetriever("data/corpus.csv")
    for doc_id, text, score in r.retrieve("How do I undo uncommitted changes temporarily?"):
        print(f"{score:.3f}  {doc_id}  {text[:70]}")
