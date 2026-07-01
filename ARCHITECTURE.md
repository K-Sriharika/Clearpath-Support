# Clearpath RAG Agent — Architecture & Design Rationale

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLEARPATH SUPPORT AGENT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Knowledge Base Files                                             │
│  ┌──────────────┐ ┌──────────────────┐ ┌──────────────────────┐ │
│  │clearpath_    │ │clearpath_        │ │clearpath_            │ │
│  │faq.txt       │ │releases.txt      │ │runbook.txt           │ │
│  │(6 Q&A pairs) │ │(4 version blocks)│ │(5 issue blocks)      │ │
│  └──────┬───────┘ └────────┬─────────┘ └──────────┬───────────┘ │
│         │                  │                        │             │
│         └──────────────────┼────────────────────────┘             │
│                            ▼                                      │
│                   ┌────────────────┐                              │
│                   │  chunker.py    │  One chunk per semantic unit: │
│                   │                │  • 1 Q&A pair per FAQ chunk  │
│                   │  15 chunks     │  • 1 version per release     │
│                   │  total         │  • 1 issue per runbook entry │
│                   └───────┬────────┘                              │
│                           │                                       │
│                           ▼                                       │
│                   ┌────────────────┐                              │
│                   │  embedder.py   │  Primary: voyage-3-lite API  │
│                   │                │  Fallback: TF-IDF (0 cost)   │
│                   │  vector_store  │  Cached to disk by content   │
│                   │  .json (disk)  │  hash — free warm starts     │
│                   └───────┬────────┘                              │
│                           │                                       │
│  Query ──────────────────►│ RETRIEVAL                             │
│                           │ Embed query → cosine similarity       │
│                           │ → top-3 chunks                        │
│                           │                                       │
│                           ▼                                       │
│                   ┌────────────────┐                              │
│                   │  agent.py      │  Confidence gate:             │
│                   │                │  top score < 0.15             │
│                   │  CONFIDENCE    │  → FALLBACK (no LLM call)    │
│                   │  GATE          │                               │
│                   └───────┬────────┘                              │
│                           │  score ≥ 0.15                         │
│                           ▼                                       │
│                   ┌────────────────┐                              │
│                   │ claude-haiku   │  System prompt:               │
│                   │ -4-5           │  • Answer ONLY from chunks   │
│                   │                │  • Return JSON with           │
│                   │ ~460 tokens    │    confident: true/false      │
│                   │ avg per query  │  • Cite [SOURCE N] labels    │
│                   └───────┬────────┘                              │
│                           │                                       │
│                    ┌──────┴──────┐                                │
│                    │             │                                 │
│              confident=true  confident=false                      │
│                    │             │                                 │
│                    ▼             ▼                                 │
│              ┌──────────┐ ┌──────────────┐                        │
│              │  Answer  │ │  Fallback    │                        │
│              │+ sources │ │  message +   │                        │
│              │          │ │  escalation  │                        │
│              └──────────┘ └──────────────┘                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design Decisions

### 1. Chunking Strategy

**Decision:** One semantic unit per chunk (one Q&A pair, one version block, one issue).

**Rationale:** The three KB documents each have natural, pre-existing semantic boundaries:
- FAQ: each Q&A is self-contained and independently answerable
- Release notes: each version block is a coherent set of changes for a product state
- Runbook: each issue block is a complete troubleshooting procedure

Splitting at these boundaries gives 15 chunks of 60–150 tokens each. Larger chunks
would lower retrieval precision (the vector represents a mix of concepts). Smaller
chunks (e.g. splitting runbook steps) would break procedural context.

**Result:** 15 chunks total — small enough that cosine search over all of them
completes in <1ms without needing an ANN index.

### 2. Embedding Model

**Primary: voyage-3-lite**

Reasons:
- Purpose-built for retrieval (outperforms general-purpose models on asymmetric retrieval tasks)
- Cheapest Anthropic-family embedding model ($0.02/M tokens)
- Single-vendor stack (Anthropic API key covers both embeddings and generation)
- Full KB indexing costs ~$0.000036; re-indexing on every cold start is still negligible

**Fallback: TF-IDF (zero cost, zero external deps)**

The TF-IDF fallback ensures the agent stays functional during API outages or in
restricted network environments. The fallback is auto-detected by checking the
stored embedding_method field and re-vectorising the query at query time.

**Known TF-IDF limitation:** Vocabulary mismatch. "Maximum date range" ≠ "90 days"
in token space. See TC-05 in the evaluation report. Voyage resolves this.

### 3. Vector Store

**Decision:** Plain JSON file on disk.

**Rationale:** At 15 chunks, even a full linear cosine scan takes <1ms. Adding
Pinecone/Weaviate would introduce network latency, additional credentials, and a
$70+/month cost floor — all for no performance benefit at this scale.

The store includes a content hash: if the KB files change, `build_index.py` detects
the change and re-embeds. If content is unchanged, the cached store is returned
immediately (zero API cost).

**Migration path:** The `build_vector_store` and `retrieve` interfaces are cleanly
separated. Swapping the JSON store for a proper vector DB requires only changing
`embedder.py` — no changes to agent.py or rag_pipeline.py.

### 4. Model Tier for Answer Generation

**Decision: claude-haiku-4-5**

Reasoning:
- The task is instruction-following over short retrieved text, not complex reasoning
- Haiku reliably follows JSON output format instructions
- At 300 queries/day × ~460 tokens prompt + ~150 tokens output:
  estimated cost ~$0.09/day at Haiku pricing
- Prompt structure keeps context window small: system (~200 tokens) +
  3 chunks (~240 tokens) + query (~20 tokens) = ~460 tokens per call

**When to upgrade to Sonnet:** If answer quality degrades on ambiguous multi-source
questions, or if the KB grows to 100+ chunks requiring more summarisation.

### 5. Two-Stage Fallback

**Stage 1 — Retrieval confidence gate (no LLM call):**
If the top cosine similarity score < 0.15, the query is considered out-of-scope
and the fallback message is returned immediately. This saves one LLM call ($0.0003)
for every unanswerable query.

**Stage 2 — Model self-assessment:**
The system prompt instructs the model to return `"confident": false` when it
cannot ground its answer in the provided context. This catches cases where the
top score passes the threshold but the retrieved chunks don't actually answer
the question (e.g. TC-02 where top-1 is a Tag/Label FAQ chunk).

**Fallback message design:** The fallback is actionable — it suggests rephrasing,
escalating, or flagging a KB gap. It does not say "I don't know" and stop.

### 6. Grounding Enforcement

The system prompt:
1. Explicitly forbids drawing on outside knowledge
2. Labels each chunk `[SOURCE N: filename chunk X]` so the model can cite them
3. Requires a structured JSON response (enables reliable `confident` parsing)
4. Instructs the model to return `confident: false` rather than speculate

### 7. Cost Controls Summary

| Decision | Annual savings at 300 q/day |
|----------|-----------------------------|
| Haiku vs Sonnet for generation | ~$1,200/year |
| Confidence gate (skip LLM for fallbacks) | ~$30/year (at 5% fallback rate) |
| Content-hash caching (skip re-embed) | ~$0.01/year (negligible but free) |
| TF-IDF fallback (skip embed on outage) | Availability, not cost |
| Top-K=3 (not 5) | ~40% fewer prompt tokens from context |

## File Structure

```
clearpath-rag/
├── knowledge_base/
│   ├── clearpath_faq.txt
│   ├── clearpath_releases.txt
│   └── clearpath_runbook.txt
├── src/
│   ├── chunker.py          # KB ingestion and chunking
│   ├── embedder.py         # Embedding + vector store + retrieval
│   ├── agent.py            # Answer generation + judge evaluation
│   ├── build_index.py      # One-time indexing script
│   └── rag_pipeline.py     # Top-level pipeline + CLI
├── evaluation/
│   ├── eval_suite.py       # 8-query evaluation harness
│   ├── eval_results.json   # Machine-readable results
│   └── eval_report.md      # Human-readable report with findings
├── vector_store.json       # Persisted embeddings (generated)
├── ARCHITECTURE.md         # This document
├── README.md               # Setup and usage
└── requirements.txt        # Dependencies
```

## Extending the System

**Adding a new KB document:**
1. Add the file to `knowledge_base/`
2. Add a `chunk_X()` function to `chunker.py`
3. Add a call to it in `ingest_knowledge_base()`
4. Run `python src/build_index.py` — vectors are rebuilt automatically

**Switching to Voyage embeddings:**
Set `ANTHROPIC_API_KEY` in the environment. The embedder detects it and uses
Voyage automatically. Re-run `build_index.py` to re-embed.

**Scaling to a larger KB (100+ documents):**
Replace the JSON store with Pinecone or Weaviate. Only `build_vector_store()`
and `retrieve()` in `embedder.py` need changing.
