# Clearpath Support Agent — Engineering Deliverable

A RAG-powered support agent for Clearpath Support Systems, built to replace the
current manual process where agents search three disconnected knowledge bases
(FAQ, release notes, runbook) and customers receive inconsistent answers when a
product update outpaces the runbook.

This repository contains all three tasks of the deliverable, each in its own folder.

## Task 1 — RAG agent implementation → [`task1-rag-agent/`](task1-rag-agent/)

The core engineering deliverable: a working RAG pipeline that chunks and embeds
the three KB documents, retrieves grounded passages, and answers strictly from
retrieved content. Includes a two-stage fallback for out-of-scope questions
(cosine-threshold gate + model self-assessment), cost-tiered model selection
(Haiku for generation, Sonnet for judging), and an 8-query LLM-as-a-judge
evaluation. A standalone `demo.html` provides an interactive UI with a live
retrieval-trace panel.

Run it:
```bash
cd task1-rag-agent
pip install -r requirements.txt
python src/build_index.py
python src/rag_pipeline.py        # interactive CLI
python evaluation/eval_suite.py   # LLM-as-judge evaluation
```

## Task 2 — Architecture diagram & rationale → [`task2-architecture/`](task2-architecture/)

A working design artifact for the engineering review: an end-to-end architecture
diagram covering ingestion, embedding, vector storage, retrieval, generation,
fallback routing, and evaluation, with labeled data flows, explicit state
boundaries, concurrency handling, and failure modes. The accompanying rationale
(one page) explains the key decisions and the single-tenant → multi-tenant
production path, specific to Clearpath's situation.

- `diagram/architecture_diagram.svg` / `.png` — the diagram
- `ARCHITECTURE_RATIONALE.md` — the written rationale

## Task 3 — Vector store, schema & retrieval integration → [`task3-vector-store/`](task3-vector-store/)

A ChromaDB-backed vector store with a schema (embedding + source document +
chunk index + raw text) that supports metadata-filtered retrieval at query
time, not as a post-processing step. Uses cosine distance (explained in
`vector_store.py`), and demonstrates both base semantic search and
source-filtered search in a runnable test script.

Run it:
```bash
cd task3-vector-store
pip install -r requirements.txt
python src/build_index.py
python tests/test_retrieval.py
```

## Shared conventions across tasks

- **Embeddings:** `voyage-3-lite` when `ANTHROPIC_API_KEY` is set; a zero-cost,
  zero-dependency TF-IDF fallback otherwise. The same fallback keeps every task
  runnable offline.
- **Chunking:** one semantic unit per chunk (one Q&A pair, one version block,
  one runbook issue), carrying `source` + `chunk_index` metadata.
- **Cost discipline:** cheap model for generation, capable model reserved for
  evaluation; retrieval capped at top-3 to bound prompt tokens.
