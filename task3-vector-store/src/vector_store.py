"""
vector_store.py — Vector database layer for the Clearpath knowledge base.

Backend: ChromaDB (persistent, local, embedded — no server, no external
account, no per-month cost floor). At Clearpath's current KB size (a
handful of documents today, plausibly hundreds as the product grows) an
embedded DB with a real ANN index and native metadata filtering is the
right tradeoff: cheaper and simpler to operate than a managed service like
Pinecone/Weaviate, but — unlike the plain-JSON store used in Task 1 —
Chroma gives us a query-time `where` filter instead of a python-side
post-filter, and an index that scales past linear scan if the KB grows.

Schema
------
Each vector in the collection carries:
  - id            : "{source}::{chunk_index}"  (stable, human-readable, unique)
  - embedding     : the chunk's dense/TF-IDF vector (computed in embedder.py)
  - document      : the raw chunk text (Chroma's built-in text field —
                     duplicated into metadata too, see below, so callers can
                     get it back from either place without a second lookup)
  - metadata:
      source        (str)  e.g. "clearpath_faq.txt"        — which KB file
      chunk_index   (int)  e.g. 2                            — position within that file
      chunk_type    (str)  "faq" | "release_notes" | "runbook"
      text          (str)  raw chunk text (same as `document`)

Storing `source` and `chunk_index` as first-class metadata fields (rather
than encoding them into the id and parsing it back out) is what makes
metadata-filtered retrieval possible: "find the best semantic match, but
only within clearpath_releases.txt" is a single `collection.query(...,
where={"source": "clearpath_releases.txt"})` call, not a retrieve-then-
discard post-processing step.

Distance metric
----------------
We configure the collection with `hnsw:space = "cosine"`.

Why cosine and not Euclidean (L2) or raw dot product:
  - Both embedding paths in embedder.py (Voyage and the TF-IDF fallback)
    L2-normalize every vector before it's stored. Once vectors are unit
    length, cosine similarity, dot product, and (monotonically) L2 distance
    all rank results identically — but cosine is the metric Voyage's model
    was optimized against, and it's the metric that stays meaningful if we
    ever mix vectors of different raw magnitudes (e.g. a future embedding
    path that isn't pre-normalized). Euclidean distance is sensitive to
    vector magnitude, which for TF-IDF specifically correlates with chunk
    length/word-count rather than topical relevance — exactly the kind of
    spurious signal cosine is designed to cancel out.
  - Support queries here are short-vs-short or short-query-vs-medium-chunk
    comparisons where we care about *directional* semantic match ("is this
    chunk about the same thing as this query"), not absolute vector
    magnitude. Cosine is the standard choice for that comparison shape.
"""

from pathlib import Path
from typing import Dict, List, Optional

import chromadb

DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "clearpath_kb"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_or_create_collection(client: chromadb.PersistentClient = None):
    """
    Recreate the collection schema from code alone — no manual console
    steps required by another engineer picking this up.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",  # see module docstring for rationale
            "description": "Clearpath support KB: FAQ + release notes + runbook chunks",
        },
    )


def index_chunks(chunks: List[Dict], vectors: List[List[float]], reset: bool = True) -> "chromadb.Collection":
    """
    Populate the collection from chunk dicts (from chunker.py) and their
    matching vectors (from embedder.py). One point per chunk.
    """
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = get_or_create_collection(client)

    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "chunk_type": c["chunk_type"],
            "text": c["text"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


def semantic_search(
    collection,
    query_vector: List[float],
    top_k: int = 3,
) -> List[Dict]:
    """
    Base semantic search: rank every chunk in the collection by cosine
    similarity to the query vector, no metadata restriction.
    """
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    return _format_results(results)


def filtered_search(
    collection,
    query_vector: List[float],
    source: Optional[str] = None,
    chunk_type: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict]:
    """
    Metadata-filtered semantic search: rank chunks by cosine similarity,
    restricted at query time (via Chroma's `where` clause, not a post-hoc
    filter) to a specific source document and/or chunk type.

    Example: "what does the release log say about webhooks" should only
    consider clearpath_releases.txt chunks, even if a runbook chunk about
    webhooks would otherwise score higher.
    """
    where = {}
    if source and chunk_type:
        where = {"$and": [{"source": source}, {"chunk_type": chunk_type}]}
    elif source:
        where = {"source": source}
    elif chunk_type:
        where = {"chunk_type": chunk_type}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where or None,
    )
    return _format_results(results)


def _format_results(raw: Dict) -> List[Dict]:
    """Flatten Chroma's batched response (we only ever send one query) into
    a simple list of {id, text, source, chunk_index, chunk_type, distance,
    similarity} dicts."""
    out = []
    ids = raw["ids"][0]
    distances = raw["distances"][0]
    metadatas = raw["metadatas"][0]
    for id_, dist, meta in zip(ids, distances, metadatas):
        out.append({
            "id": id_,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "chunk_type": meta["chunk_type"],
            "text": meta["text"],
            "distance": dist,
            # Chroma's cosine "distance" is 1 - cosine_similarity for
            # normalized vectors; convert back to a 0-1 similarity score
            # for readability in the demo output.
            "similarity": 1 - dist,
        })
    return out
