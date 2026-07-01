# Clearpath Support RAG Agent

A RAG-powered support agent that answers questions grounded in the Clearpath
knowledge base (FAQ, release notes, runbook), with fallback handling for
out-of-scope queries and LLM-as-judge evaluation.

## Quick Start

```bash
# 1. Install dependencies
pip install numpy anthropic

# 2. Set your API key (optional — TF-IDF fallback works without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the vector index
python src/build_index.py

# 4. Ask a question (CLI)
python src/rag_pipeline.py

# 5. Run the evaluation suite
python evaluation/eval_suite.py
```

## Architecture

See `ARCHITECTURE.md` for the full design rationale.

## Cost Model

| Operation | Cost |
|-----------|------|
| KB indexing (Voyage, one-time) | ~$0.000036 |
| Per query (Haiku generation) | ~$0.0003 |
| 300 queries/day | ~$0.09/day |
| LLM judge eval (Sonnet, 8 calls) | ~$0.02 one-time |

## Key Design Choices

- **Chunk by semantic unit:** one Q&A pair, one version block, one issue
- **voyage-3-lite** for embeddings (TF-IDF fallback if API unavailable)
- **claude-haiku-4-5** for generation (cost-optimised; Sonnet for judge)
- **Two-stage fallback:** cosine threshold gate + model self-assessment
- **Content hash caching:** KB re-embedded only when files change

## Files

```
src/chunker.py        — KB ingestion and chunking
src/embedder.py       — Embedding, vector store, retrieval
src/agent.py          — Answer generation + judge evaluation
src/build_index.py    — Build/refresh the vector store
src/rag_pipeline.py   — Top-level pipeline + interactive CLI
evaluation/           — Eval harness and results
knowledge_base/       — The three source KB files
ARCHITECTURE.md       — Full design rationale and diagrams
```
