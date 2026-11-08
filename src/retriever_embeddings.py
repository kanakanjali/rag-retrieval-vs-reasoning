"""
retriever_embeddings.py — dense semantic retrieval using sentence-transformers.

Unlike TF-IDF, this embeds queries and documents into a shared vector space
trained to capture meaning, not just word overlap, so it can match
"undo uncommitted changes temporarily" to a doc that says "shelves... reapplies
them later" even with almost no literal word overlap.

Requires: pip install sentence-transformers
(Needs internet the first time, to download the model weights — after that
it's cached locally.)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    def __init__(self, corpus_csv_path, text_col="text", id_col="doc_id",
                 model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(corpus_csv_path)
        self.ids = self.df[id_col].tolist()
        self.texts = self.df[text_col].tolist()
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(self.texts, normalize_embeddings=True)

    def retrieve(self, query, top_k=3):
        """Return list of (doc_id, text, score) for the top_k most similar docs."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.doc_embeddings @ q_emb  # cosine sim, since both are normalized
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in ranked_idx]


if __name__ == "__main__":
    # quick manual sanity check
    r = EmbeddingRetriever("data/corpus.csv")
    for doc_id, text, score in r.retrieve("How do I undo uncommitted changes temporarily?"):
        print(f"{score:.3f}  {doc_id}  {text[:70]}")
