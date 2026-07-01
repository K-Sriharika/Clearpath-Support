# Clearpath Support Agent — Architecture Rationale
**Task 2 of 3 · Engineering Review Draft**

See `architecture_diagram.svg` for the full component and data-flow diagram.

## Why the components are separated this way

Ingestion, retrieval, generation, and evaluation are split into independent modules rather than one script because each has a different **change frequency** and a different **cost profile**. The knowledge base (three flat text files) changes on a sprint cadence, so ingestion — chunking and embedding — runs offline and writes to a persisted vector store; it never runs in the request path. The query path (embed → retrieve → generate) runs synchronously per agent question and must stay fast and cheap, since it runs hundreds of times a day. Evaluation runs separately again, on demand, because it uses a more expensive judge model and isn't something a support agent should ever wait on. Keeping these as separate modules means each can be scaled, cached, or replaced without touching the others — for example, swapping the embedding backend from TF-IDF to Voyage required no changes to the retriever or generator.

## Key trade-offs

**Two-stage fallback over a single confidence check.** A cosine-similarity gate runs before any LLM call, so genuinely out-of-scope questions (e.g. "how do I integrate with Slack?") never reach the generator — this is the primary cost lever, since it's the difference between a free vector lookup and a paid API call. A second gate, the model's own structured `confident` field, catches cases where retrieval returned *plausible but wrong* chunks — something a similarity score alone can't detect. The trade-off is added latency and complexity versus a single fixed threshold; we judged the reduction in hallucinated answers (Clearpath's core pain point) worth it.

**Cost-tiered models.** Haiku generates every answer; Sonnet is reserved for judge-only evaluation. This roughly matches model capability to task difficulty — generation is grounded, single-document lookup in most cases, while judging correctness across nuanced version history benefits from a stronger model. At Clearpath's ~300 tickets/day volume, this keeps per-query cost predictable while still catching quality regressions in eval.

**Top-K = 3 retrieval.** Caps prompt tokens and cost per query, but caps recall too — a question that legitimately needs four chunks (e.g. spanning FAQ, two release notes, and the runbook) will lose the weakest match. This was accepted for the prototype and is the first knob to revisit if eval shows recall-related failures.

**File-backed vector store, no session state.** The store is a single persisted index rebuilt from source text files; the source-of-truth is always the flat files, not the index. Making the query path fully stateless (no server-held conversation memory) was a deliberate choice — it costs nothing for a single-turn Q&A prototype and it's what makes horizontal scaling of the *query* path straightforward later, even though the *store* itself doesn't scale horizontally yet.

## Single-tenant prototype → multi-tenant production

Three things would need to change, in priority order:

1. **Tenant isolation in the vector store.** The single biggest change. A file-backed index with no tenant boundary is fine for one customer's three documents; it is not safe once multiple customers' knowledge bases live in the same system — a retrieval bug must never surface Customer A's runbook to Customer B's agent. This means moving to a vector store with a tenant-scoped namespace or partition key (e.g. pgvector with a `tenant_id` column, or per-tenant collections in a managed vector DB), and enforcing tenant scoping at the query layer, not just in application logic.
2. **Shared, tenant-aware caching.** The content-hash disk cache works for one process; at scale it should move to a shared cache (e.g. Redis) keyed by `tenant_id + content_hash`, so cache hits are safe across tenants without leaking content between them.
3. **Decoupling the query path from ingestion/eval processes**, adding a request queue and autoscaling in front of the generator so a burst from one tenant doesn't starve another, plus per-tenant rate limiting and cost tracking so usage-based pricing (or abuse) is visible per customer rather than aggregated.

Chunking, the two-stage fallback, and the cost-tiered model choice all carry forward unchanged — they're tenant-agnostic decisions.
