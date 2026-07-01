# Clearpath Vector Store — Architecture & Rationale

Task 3 of 3: vector store setup, schema design, and retrieval integration.

See `architecture_diagram.svg` in this folder for the visual flow: KB files
→ `chunker.py` → `embedder.py` → Chroma collection → `semantic_search()` /
`filtered_search()` → ranked results.

## Why a real vector database here, not the JSON file from Task 1

Task 1 used a plain JSON file as the vector store, which was the right call
at 15 chunks with no filtering requirement — a linear cosine scan over 15
vectors takes under a millisecond, and Pinecone/Weaviate would have added
network latency and a monthly cost floor for zero benefit.

Task 3's requirement changes the calculus: **metadata filtering has to
happen at query time, not as a post-processing step**, and the schema has
to be something another engineer can recreate from code alone. A hand-rolled
JSON store can do this too (loop over the list, check `chunk["source"] ==
target` before or after scoring), but that's re-implementing exactly what a
vector database's `where` clause and index already do — and it stops
working cleanly once the KB grows past "fits in memory, scan is instant."

**Decision: ChromaDB**, embedded/persistent mode (`chromadb.PersistentClient`).
No server process, no external account, no cost — but it's a real vector
database with an ANN index (HNSW) and native metadata filtering, so the
implementation here is representative of what scaling to a managed service
(Pinecone, Weaviate, pgvector) would look like: same `where`-clause query
pattern, same schema shape, different `get_client()` implementation.

## Schema

Each point in the `clearpath_kb` collection:

| Field | Type | Example | Purpose |
|---|---|---|---|
| `id` | str | `"clearpath_faq.txt::4"` | Stable, human-readable, unique per chunk |
| `embedding` | float[] | `[0.021, -0.114, ...]` | The chunk's vector (Voyage or TF-IDF) |
| `document` | str | full chunk text | Chroma's built-in text field |
| `metadata.source` | str | `"clearpath_faq.txt"` | Which KB file — filterable |
| `metadata.chunk_index` | int | `4` | Position within that file |
| `metadata.chunk_type` | str | `"faq"` | `faq` \| `release_notes` \| `runbook` — filterable |
| `metadata.text` | str | full chunk text | Duplicated into metadata so it comes back with any query result without a second lookup |

`source` and `chunk_type` are stored as first-class metadata fields
specifically so they're usable in a `where` clause at query time — encoding
them into the `id` string and parsing it back out afterward would force a
post-hoc filter instead, which is the thing this task asks us to avoid.

Full schema definition lives in `src/vector_store.py::get_or_create_collection()`
— running `python src/build_index.py` recreates the collection from that
code with no manual console steps.

## Distance metric: cosine

Configured via `collection_metadata={"hnsw:space": "cosine"}` at creation
time.

Why cosine over Euclidean (L2) or raw dot product:

- Every vector this project produces (`embedder.py`, both the Voyage path
  and the TF-IDF fallback) is L2-normalized before storage. Once vectors
  are unit length, cosine similarity, dot product, and (monotonically) L2
  distance rank results *identically* — but cosine is the metric Voyage's
  embedding model was trained against, and it's the metric that stays
  correct if a future embedding path isn't pre-normalized.
- For the TF-IDF path specifically, raw vector magnitude correlates with
  chunk word count, not topical relevance — a long runbook entry would
  otherwise out-score a short, more relevant FAQ answer under Euclidean
  distance purely because it has a bigger vector. Cosine cancels that out
  by construction.
- Support queries here are short-query-vs-chunk comparisons where the
  question is "does this chunk point in the same semantic direction as the
  query," not "is this chunk's vector close in absolute position." That's
  exactly the comparison cosine is built for.

## Retrieval modes implemented

### 1. Base semantic search — `vector_store.semantic_search()`
Ranks every chunk in the collection by cosine similarity to the query
vector. No restriction. This is what you want for "what's the answer to
this support question," full stop.

### 2. Metadata-filtered search — `vector_store.filtered_search()`
Same ranking, restricted at query time via Chroma's `where` clause to a
specific `source` and/or `chunk_type`. This is what you want for "what does
the *release log specifically* say about webhook retries" — a case the
brief calls out explicitly, since the runbook also mentions webhook retries
and could otherwise out-rank the release note that's actually the
authoritative source for that behavior.

Both modes are demonstrated together in `tests/test_retrieval.py`, including
a direct A/B check (Test 4) showing the filtered search returns a source
document that the unfiltered top-1 result would not have surfaced.

## Query-time embedding: a caveat worth documenting

The TF-IDF fallback path has no out-of-vocabulary generalization — it's a
fixed vocabulary fitted once on the corpus at index-build time. A query
containing a word the corpus never used projects to a zero weight for that
term, not an error, but also not a real semantic match. `embedder.py`
persists the fitted vocabulary/IDF weights to `tfidf_model.json` at index
time specifically so query-time embedding (`embed_query_text()`) stays in
the exact same vector space as the stored chunks — re-fitting per query
would silently change the store's dimensionality and break lookups (this
was caught and fixed during implementation; see `embedder.py`'s
`TfidfModel` class for the corrected approach).

This limitation goes away entirely once `ANTHROPIC_API_KEY` is set and the
Voyage path is used — Voyage embeds arbitrary query text directly, no
corpus-fitted vocabulary required. The same `embed_query_text()` interface
covers both paths so callers don't need to know which one is active.

## Recreating this store on another engineer's machine

```bash
pip install chromadb numpy
python src/build_index.py     # chunk KB → embed → create/populate collection
python tests/test_retrieval.py  # demonstrates both retrieval modes
```

No manual collection setup, no console clicking — the schema (including
the cosine distance-metric configuration) is defined entirely in
`src/vector_store.py::get_or_create_collection()`.

## Scaling path

- **Bigger KB (100s–1000s of chunks):** no code change needed — Chroma's
  HNSW index handles this natively; `semantic_search()`/`filtered_search()`
  keep working as-is.
- **Managed vector DB (Pinecone/Weaviate/pgvector) instead of embedded
  Chroma:** only `get_client()` and `get_or_create_collection()` change.
  `semantic_search()`, `filtered_search()`, and every caller stay the same
  — this is the same separation-of-concerns decision documented in Task 1's
  `ARCHITECTURE.md` for swapping the embedder.
- **Real dense embeddings everywhere:** set `ANTHROPIC_API_KEY` and re-run
  `build_index.py` — `embedder.py` auto-detects the key and switches from
  TF-IDF to `voyage-3-lite` with no changes to `vector_store.py` or the
  test script.
