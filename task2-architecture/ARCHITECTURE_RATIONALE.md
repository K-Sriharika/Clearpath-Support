# Clearpath Support Agent — Architecture Rationale
**Task 2 of 3 · Engineering Review Draft**

See `architecture_diagram.svg` for the full component and data-flow diagram.

## Why the components are separated this way

Ingestion, retrieval, generation, and evaluation are independent modules because each has a different change frequency and cost profile. The knowledge base changes on a sprint cadence, so ingestion (chunking + embedding) runs offline into a persisted vector store — never in the request path. The query path (embed → retrieve → generate) runs synchronously per agent question and must stay fast and cheap, since it runs hundreds of times a day. Evaluation runs separately again, on demand, using a more expensive judge model that a support agent should never wait on. This separation lets each piece scale, cache, or get replaced independently — swapping the embedding backend from TF-IDF to Voyage required no change to the retriever or generator.

## Key trade-offs

**Two-stage fallback over a single confidence check.** A cosine-similarity gate runs before any LLM call, so out-of-scope questions never reach the generator — the primary cost lever, since it's the difference between a free vector lookup and a paid API call. A second gate, the model's own structured `confident` field, catches cases where retrieval returned plausible-but-wrong chunks, which a similarity score alone can't detect. The cost is added latency and complexity versus one fixed threshold; we judged the reduction in hallucinated answers — Clearpath's core pain point — worth it.

**Cost-tiered models.** Haiku generates every answer; Sonnet is reserved for judge-only evaluation, matching model capability to task difficulty. Generation is grounded, mostly single-document lookup; judging correctness across nuanced version history benefits from a stronger model. At ~300 tickets/day, this keeps per-query cost predictable while still catching quality regressions in eval.

**Top-K = 3 retrieval.** Caps prompt tokens and cost per query, but caps recall too — a question needing four chunks across documents will lose its weakest match. Accepted for the prototype; first knob to revisit if eval shows recall failures.

**File-backed store, no session state.** The vector store is a persisted index rebuilt from the flat text files, which remain the source of truth. The query path holds no server-side conversation memory — free for a single-turn prototype, and what makes horizontal scaling of the query path straightforward later, even though the store itself doesn't scale horizontally yet.

## Single-tenant prototype → multi-tenant production

Three changes, in priority order:

1. **Tenant isolation in the vector store** — the biggest change. A file-backed index with no tenant boundary is fine for one customer's documents; unsafe once multiple customers' knowledge bases share the system, since a retrieval bug must never surface Customer A's runbook to Customer B's agent. Requires a tenant-scoped namespace or partition key (e.g. pgvector with a `tenant_id` column, or per-tenant collections in a managed vector DB), enforced at the query layer.
2. **Shared, tenant-aware caching** — the content-hash disk cache works for one process; at scale it moves to a shared cache (e.g. Redis) keyed by `tenant_id + content_hash`, so hits stay safe across tenants.
3. **Decoupling the query path from ingestion/eval**, with a request queue and autoscaling in front of the generator so one tenant's burst doesn't starve another, plus per-tenant rate limiting and cost tracking.

Chunking, the two-stage fallback, and the cost-tiered model choice all carry forward unchanged — they're tenant-agnostic decisions.
