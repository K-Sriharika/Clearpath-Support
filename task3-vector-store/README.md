# Clearpath Vector Store (Task 3 of 3)

A ChromaDB-backed vector store for the Clearpath support knowledge base
(FAQ, release notes, runbook), with a schema that supports metadata-filtered
retrieval alongside base semantic search.

Full design rationale — schema, distance-metric choice, retrieval modes,
scaling path — is in `docs/ARCHITECTURE.md`. Visual architecture diagram in
`docs/architecture_diagram.svg`.

## Quick start

```bash
pip install -r requirements.txt

# (optional) use real Voyage embeddings instead of the TF-IDF fallback
export ANTHROPIC_API_KEY=sk-ant-...

# Build the collection from the three KB text files
python src/build_index.py

# Demonstrate both base semantic search and metadata-filtered search
python tests/test_retrieval.py
```

## Project structure

```
clearpath-vectorstore/
├── knowledge_base/
│   ├── clearpath_faq.txt
│   ├── clearpath_releases.txt
│   └── clearpath_runbook.txt
├── src/
│   ├── chunker.py        # semantic-unit chunking (one Q&A, one version, one issue)
│   ├── embedder.py        # Voyage primary / TF-IDF fallback, both L2-normalized
│   ├── vector_store.py    # Chroma collection: schema, cosine metric, both search modes
│   └── build_index.py     # chunk → embed → index, run once (or on KB change)
├── tests/
│   └── test_retrieval.py  # demonstrates + asserts on both retrieval modes
├── docs/
│   ├── ARCHITECTURE.md
│   └── architecture_diagram.svg
└── requirements.txt
```

## What "metadata-filtered" means here

`vector_store.filtered_search(collection, query_vector, source="clearpath_releases.txt")`
restricts the similarity search to a single source document **at query
time**, via Chroma's `where` clause — not by retrieving broadly and
discarding results afterward. This is what lets an agent ask "what does the
release log say about webhook retries specifically" and get an answer
grounded in the release notes even though the runbook also mentions webhook
retries and could otherwise rank higher.
