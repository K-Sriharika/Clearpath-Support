"""
embedder.py — Embedding generation and vector store management.

Design rationale:
  - Primary: Anthropic Voyage embeddings (voyage-3-lite).
    Cost: $0.02/M tokens. Full KB (~1,800 tokens) = $0.000036 once.
    We cache to disk and only re-embed when KB content changes.
  - Fallback: TF-IDF bag-of-words (zero cost, zero external deps).
    Used automatically if the Voyage API is unavailable.
  - Similarity: cosine similarity over numpy arrays is fast enough
    for ~50 chunks with no need for an ANN index.
"""

import json
import hashlib
import os
import math
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

VECTOR_STORE_PATH = Path(__file__).parent.parent / "vector_store.json"


def _tfidf_vectors(texts: List[str]) -> List[List[float]]:
    """Pure-Python TF-IDF — no external deps needed."""
    def tokenise(t):
        return re.findall(r'[a-z0-9]+', t.lower())

    tokenised = [tokenise(t) for t in texts]
    vocab = sorted(set(tok for toks in tokenised for tok in toks))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = len(texts)

    # TF
    tf_matrix = []
    for toks in tokenised:
        vec = [0.0] * V
        for tok in toks:
            if tok in vocab_idx:
                vec[vocab_idx[tok]] += 1
        total = sum(vec) or 1
        tf_matrix.append([x / total for x in vec])

    # IDF
    idf = []
    for j in range(V):
        df = sum(1 for row in tf_matrix if row[j] > 0)
        idf.append(math.log((N + 1) / (df + 1)) + 1)

    # TF-IDF
    tfidf = [[tf_matrix[i][j] * idf[j] for j in range(V)] for i in range(N)]
    return tfidf, vocab


def _embed_voyage(texts: List[str]) -> List[List[float]]:
    """Call Anthropic Voyage embeddings endpoint."""
    import urllib.request
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    url = "https://api.anthropic.com/v1/embeddings"
    payload = json.dumps({
        "model": "voyage-3-lite",
        "input": texts,
        "input_type": "document"
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return [item["embedding"] for item in data["data"]]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _content_hash(chunks: List[Dict]) -> str:
    blob = json.dumps([c["text"] for c in chunks], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def build_vector_store(chunks: List[Dict], force_rebuild: bool = False) -> Dict:
    """
    Embed all chunks and persist to disk.
    Skips re-embedding if content hash matches the cached store.
    Tries Voyage first; falls back to TF-IDF.
    """
    new_hash = _content_hash(chunks)

    if VECTOR_STORE_PATH.exists() and not force_rebuild:
        with open(VECTOR_STORE_PATH) as f:
            store = json.load(f)
        if store.get("content_hash") == new_hash:
            print(f"[embedder] Cache hit (hash {new_hash[:8]}). Skipping re-embed.")
            return store

    print(f"[embedder] Building vector store for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]

    # Try Voyage, fall back to TF-IDF
    vocab = None
    try:
        vectors = _embed_voyage(texts)
        method = "voyage-3-lite"
        print(f"[embedder] Used Voyage embeddings (dim={len(vectors[0])})")
    except Exception as e:
        print(f"[embedder] Voyage unavailable ({e}); using TF-IDF fallback.")
        vectors, vocab = _tfidf_vectors(texts)
        method = "tfidf-fallback"
        print(f"[embedder] TF-IDF vocab size: {len(vocab)}")

    store = {
        "content_hash": new_hash,
        "embedding_method": method,
        "vocab": vocab,  # only populated for tfidf-fallback
        "chunks": [
            {**chunk, "vector": vec}
            for chunk, vec in zip(chunks, vectors)
        ]
    }
    with open(VECTOR_STORE_PATH, "w") as f:
        json.dump(store, f)
    print(f"[embedder] Store saved to {VECTOR_STORE_PATH}")
    return store


def load_vector_store() -> Dict:
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError("Vector store not found. Run build_index.py first.")
    with open(VECTOR_STORE_PATH) as f:
        return json.load(f)


def retrieve(query: str, store: Dict, top_k: int = 3) -> List[Tuple[float, Dict]]:
    """
    Embed the query using the same method as the store, then rank by cosine sim.
    Returns [(score, chunk), ...] sorted descending.
    """
    method = store.get("embedding_method", "tfidf-fallback")

    if method == "tfidf-fallback":
        # Re-build TF-IDF space with query appended to ensure same vocab
        all_texts = [c["text"] for c in store["chunks"]] + [query]
        all_vecs, _ = _tfidf_vectors(all_texts)
        query_vec = all_vecs[-1]
        chunk_vecs = all_vecs[:-1]
        results = [
            (cosine_similarity(query_vec, chunk_vecs[i]), store["chunks"][i])
            for i in range(len(store["chunks"]))
        ]
    else:
        # Voyage: embed query as a single short text
        query_vecs = _embed_voyage([query])
        query_vec = query_vecs[0]
        results = [
            (cosine_similarity(query_vec, chunk["vector"]), chunk)
            for chunk in store["chunks"]
        ]

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
