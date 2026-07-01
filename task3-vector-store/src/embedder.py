"""
embedder.py — Turns chunk text into vectors.

Primary: Anthropic Voyage embeddings (voyage-3-lite) when ANTHROPIC_API_KEY
is set. Purpose-built for retrieval, cheap ($0.02/M tokens), single-vendor
stack with the generation model used elsewhere in this project.

Fallback: pure-Python TF-IDF. Zero cost, zero external dependencies, used
automatically when no API key is available (e.g. this offline demo run).
Both paths produce L2-normalized vectors so that cosine similarity and
dot-product similarity are equivalent — see vector_store.py for why that
matters for the distance-metric choice.
"""

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _tokenize(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def _tf_row(toks: List[str], vocab_idx: Dict[str, int], V: int) -> List[float]:
    vec = [0.0] * V
    for tok in toks:
        idx = vocab_idx.get(tok)
        if idx is not None:  # out-of-vocabulary tokens are dropped, same
            vec[idx] += 1     # as any TF-IDF model at query time
    total = sum(vec) or 1
    return [x / total for x in vec]


def fit_tfidf(texts: List[str]) -> "TfidfModel":
    """
    Fit vocabulary + IDF weights on the corpus (the KB chunks). This is
    the *fixed* space every later query gets projected into — vocabulary
    is not allowed to grow at query time, which is what makes the stored
    vector dimensionality stable and queryable against a persisted index.
    """
    tokenized = [_tokenize(t) for t in texts]
    vocab = sorted(set(tok for toks in tokenized for tok in toks))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = len(texts)

    tf_matrix = [_tf_row(toks, vocab_idx, V) for toks in tokenized]

    idf = []
    for j in range(V):
        df = sum(1 for row in tf_matrix if row[j] > 0)
        idf.append(math.log((N + 1) / (df + 1)) + 1)

    return TfidfModel(vocab=vocab, vocab_idx=vocab_idx, idf=idf)


class TfidfModel:
    """Fitted TF-IDF vocabulary + IDF weights, reusable to embed new
    (query-time) texts into the same fixed vector space as the corpus."""

    def __init__(self, vocab: List[str], vocab_idx: Dict[str, int], idf: List[float]):
        self.vocab = vocab
        self.vocab_idx = vocab_idx
        self.idf = idf
        self.dim = len(vocab)

    def transform(self, texts: List[str]) -> List[List[float]]:
        rows = []
        for t in texts:
            toks = _tokenize(t)
            tf = _tf_row(toks, self.vocab_idx, self.dim)
            weighted = [tf[j] * self.idf[j] for j in range(self.dim)]
            vec = np.array(weighted, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec.tolist())
        return rows

    def to_dict(self) -> Dict:
        return {"vocab": self.vocab, "idf": self.idf}

    @classmethod
    def from_dict(cls, d: Dict) -> "TfidfModel":
        vocab = d["vocab"]
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        return cls(vocab=vocab, vocab_idx=vocab_idx, idf=d["idf"])


def _tfidf_vectors(texts: List[str]):
    """Fit a TF-IDF model on `texts` and return (vectors_for_texts, model).
    Callers that need to embed later query text MUST reuse the returned
    model's .transform() rather than re-fitting — see fit_tfidf() docstring.
    """
    model = fit_tfidf(texts)
    return model.transform(texts), model


def _embed_voyage(texts: List[str]) -> List[List[float]]:
    """Call Anthropic's Voyage embeddings endpoint."""
    import anthropic

    client = anthropic.Anthropic()
    result = client.beta.messages.embeddings.create(
        model="voyage-3-lite",
        input=texts,
    )
    vectors = [np.array(e.embedding, dtype=np.float32) for e in result.data]
    normed = [(v / np.linalg.norm(v)).tolist() for v in vectors]
    return normed


TFIDF_MODEL_PATH = Path(__file__).parent.parent / "tfidf_model.json"


def embed_texts(texts: List[str], use_api: bool = None) -> Tuple[List[List[float]], str]:
    """
    Embed a list of texts (a full corpus, e.g. at index-build time).
    Returns (vectors, method_name). When TF-IDF fallback is used, the
    fitted vocabulary/IDF model is persisted to disk so query-time text
    can later be projected into the exact same vector space — see
    embed_query_text().

    use_api=None -> auto-detect via ANTHROPIC_API_KEY presence.
    """
    if use_api is None:
        use_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if use_api:
        try:
            return _embed_voyage(texts), "voyage-3-lite"
        except Exception as e:
            print(f"[embedder] Voyage embedding failed ({e}), falling back to TF-IDF.")

    vectors, model = _tfidf_vectors(texts)
    import json
    with open(TFIDF_MODEL_PATH, "w") as f:
        json.dump(model.to_dict(), f)
    return vectors, "tfidf-fallback"


def embed_query_text(query: str, method: str) -> List[float]:
    """
    Embed a single query at retrieval time, using whichever method the
    store was built with.

    - voyage-3-lite: queries embed independently, no corpus needed.
    - tfidf-fallback: the query MUST be projected into the fitted
      vocabulary saved by embed_texts() during indexing — tokens the
      corpus never saw are simply dropped, exactly like at inference
      time for any TF-IDF model.
    """
    if method == "voyage-3-lite":
        vectors = _embed_voyage([query])
        return vectors[0]
    import json
    if not TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No persisted TF-IDF model found — run build_index.py first."
        )
    with open(TFIDF_MODEL_PATH) as f:
        model = TfidfModel.from_dict(json.load(f))
    return model.transform([query])[0]
