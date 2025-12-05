import os
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CORPUS_DIR = "../corpus"  # Now points to the unified directory


class SimpleSearchEngine:
    def __init__(self, corpus_dir: str = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.documents: List[Dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_tfidf = None

    def _load_documents(self):
        docs = []
        doc_id = 0

        if not os.path.exists(self.corpus_dir):
            print(f"Warning: Corpus directory '{self.corpus_dir}' does not exist.")
            return

        for root, _, files in os.walk(self.corpus_dir):
            for fname in files:
                if not fname.endswith(".json"):
                    continue

                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract TF-IDF specific text
                    text = data.get("tf_idf_text", "").strip()
                    url = data.get("url", "")
                    title = data.get("title", os.path.splitext(fname)[0])

                    if not text:
                        continue

                    docs.append({
                        "id": doc_id,
                        "title": title,
                        "url": url,
                        "path": path,
                        "text": text,  # Used for indexing
                    })
                    doc_id += 1

                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

        self.documents = docs
        print(f"Loaded {len(self.documents)} documents from {self.corpus_dir}")

    def build_index(self):
        self._load_documents()
        texts = [doc["text"] for doc in self.documents]

        if not texts:
            print("No documents found to index.")
            return

        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"\b\w+\b")
        self.doc_tfidf = self.vectorizer.fit_transform(texts)
        print("TF-IDF index built.")

    def search(self, query: str, top_k: int = 5):
        if self.vectorizer is None or self.doc_tfidf is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vec, self.doc_tfidf).flatten()
        top_idx = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            score = float(similarities[idx])
            doc = self.documents[idx]
            # Use the raw text snippet for display if 'text' is too processed/unreadable,
            # or just use the processed text.
            # Note: doc['text'] here is the 'tf_idf_text'.
            snippet = doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")

            results.append({
                "score": score,
                "title": doc["title"],
                "url": doc["url"],
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

        try:
            hits = engine.search(query, top_k=5)
            print(f"\nTop results for: {query!r}")
            for i, h in enumerate(hits, start=1):
                print(f"\n[{i}] {h['title']} (score={h['score']:.4f})")
                print(f"    Source: {h['url']}")
                print(f"    Snippet: {h['snippet']}")
        except RuntimeError as e:
            print(e)