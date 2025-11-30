import os
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # cosine similarity for normalized TF-IDF

CORPUS_DIR = "../corpus_tf"

class SimpleSearchEngine:
    def __init__(self, corpus_dir: str = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.documents: List[Dict] = []  # each: {"id", "title", "path", "text"}
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_tfidf = None  # sparse matrix

    def _load_documents(self):
        docs = []
        doc_id = 0

        for root, _, files in os.walk(self.corpus_dir):
            for fname in files:
                if not fname.endswith(".txt"):
                    continue

                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

                if not text:
                    continue

                title = os.path.splitext(fname)[0]
                docs.append({
                    "id": doc_id,
                    "title": title,
                    "path": path,
                    "text": text,
                })
                doc_id += 1

        self.documents = docs
        print(f"Loaded {len(self.documents)} documents from {self.corpus_dir}")

    def build_index(self):
        """
        Load documents from disk and build a TF-IDF index.
        Call this once before doing any searches.
        """
        self._load_documents()
        texts = [doc["text"] for doc in self.documents]

        # You can tune vectorizer params later (ngram_range, min_df, etc.)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\b\w+\b"
        )
        self.doc_tfidf = self.vectorizer.fit_transform(texts)
        print("TF-IDF index built.")

    def search(self, query: str, top_k: int = 5):
        """
        Return top_k most relevant documents for the query.
        """
        if self.vectorizer is None or self.doc_tfidf is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.vectorizer.transform([query])
        # linear_kernel on normalized tf-idf = cosine similarity
        similarities = linear_kernel(query_vec, self.doc_tfidf).flatten()

        # get indices of top_k scores
        top_idx = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            score = float(similarities[idx])
            doc = self.documents[idx]
            # Simple snippet: first 200 characters
            snippet = doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")
            results.append({
                "score": score,
                "title": doc["title"],
                "path": doc["path"],
                "snippet": snippet,
            })
        return results


if __name__ == "__main__":
    engine = SimpleSearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index()

    while True:
        query = input("\nEnter your query (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        hits = engine.search(query, top_k=5)
        print(f"\nTop results for: {query!r}")
        for i, h in enumerate(hits, start=1):
            print(f"\n[{i}] {h['title']} (score={h['score']:.4f})")
            print(f"Path: {h['path']}")
            print(f"Snippet: {h['snippet']}")
