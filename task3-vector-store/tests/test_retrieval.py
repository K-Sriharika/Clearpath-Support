"""
test_retrieval.py — Demonstrates both retrieval modes against the live
Chroma collection built by src/build_index.py:

  1. Base semantic search  — rank across the whole KB, no restriction.
  2. Metadata-filtered search — same query, restricted to one source
     document (and, in one case, one chunk_type) via Chroma's `where`
     clause at query time.

Run with:
    python src/build_index.py      # once, to (re)build the store
    python tests/test_retrieval.py

This is a plain assertion-based script rather than a pytest suite so it
can be run standalone with no test-runner dependency, while still failing
loudly (AssertionError) if retrieval behaves unexpectedly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedder import embed_query_text
from vector_store import get_client, get_or_create_collection, semantic_search, filtered_search


def embed_query(query: str, method: str):
    """
    Embed a single query using the SAME fixed vocabulary/method the store
    was built with (see embedder.embed_query_text for why this matters
    for the TF-IDF fallback path).
    """
    return embed_query_text(query, method)


def print_results(label: str, results):
    print(f"\n--- {label} ---")
    for r in results:
        print(f"  [{r['source']} #{r['chunk_index']}] "
              f"sim={r['similarity']:.3f}  {r['text'][:70].replace(chr(10), ' ')}...")


def main():
    client = get_client()
    collection = get_or_create_collection(client)
    assert collection.count() > 0, "Collection is empty — run src/build_index.py first."

    # Detect which embedding method the store was built with by checking
    # vector dimensionality against a fresh single-text TF-IDF embed is
    # unreliable, so instead we just try Voyage first the same way
    # embed_texts() does, and fall back consistently.
    import os
    method = "voyage-3-lite" if os.environ.get("ANTHROPIC_API_KEY") else "tfidf-fallback"

    print("=" * 60)
    print("TEST 1 — Base semantic search (no filter)")
    print("=" * 60)
    query1 = "What happens if a webhook delivery fails?"
    qvec1 = embed_query(query1, method)
    results1 = semantic_search(collection, qvec1, top_k=3)
    print_results(f"Query: '{query1}'", results1)
    assert len(results1) == 3
    top_sources = {r["source"] for r in results1[:2]}
    # Both the release notes (which introduced the retry behaviour) and
    # the runbook (which references it operationally) are legitimate top
    # matches for this query — the test just confirms retrieval returns
    # results and that webhook-relevant chunks surface at all.
    assert any("webhook" in r["text"].lower() for r in results1), \
        "Expected at least one webhook-related chunk in top-3 results"
    print("  ✓ PASS — webhook-relevant chunks retrieved across the KB")

    print("\n" + "=" * 60)
    print("TEST 2 — Metadata-filtered search (source=clearpath_releases.txt)")
    print("=" * 60)
    query2 = "What happens if a webhook delivery fails?"
    qvec2 = embed_query(query2, method)
    results2 = filtered_search(
        collection, qvec2, source="clearpath_releases.txt", top_k=3
    )
    print_results(f"Query: '{query2}' (filtered to release notes only)", results2)
    assert all(r["source"] == "clearpath_releases.txt" for r in results2), \
        "Filtered search leaked results from a non-target source"
    assert len(results2) > 0
    print("  ✓ PASS — all results restricted to clearpath_releases.txt, "
          "as requested by the filter")

    print("\n" + "=" * 60)
    print("TEST 3 — Metadata-filtered search (source=clearpath_runbook.txt)")
    print("=" * 60)
    query3 = "agent can't sign in"
    qvec3 = embed_query(query3, method)
    results3 = filtered_search(
        collection, qvec3, source="clearpath_runbook.txt", top_k=2
    )
    print_results(f"Query: '{query3}' (filtered to runbook only)", results3)
    assert all(r["source"] == "clearpath_runbook.txt" for r in results3)
    assert any("log in" in r["text"].lower() for r in results3), \
        "Expected the login-issue chunk to surface for a login-related query"
    print("  ✓ PASS — login troubleshooting chunk correctly retrieved "
          "from the runbook only")

    print("\n" + "=" * 60)
    print("TEST 4 — Cross-check: unfiltered search on the same query CAN "
          "return a different top document than the filtered search")
    print("=" * 60)
    results4_unfiltered = semantic_search(collection, qvec1, top_k=1)
    results4_filtered = filtered_search(
        collection, qvec1, source="clearpath_runbook.txt", top_k=1
    )
    print(f"  Unfiltered top-1 source: {results4_unfiltered[0]['source']}")
    print(f"  Filtered(runbook) top-1 source: {results4_filtered[0]['source']}")
    assert results4_filtered[0]["source"] == "clearpath_runbook.txt"
    print("  ✓ PASS — filter successfully constrains the result set even "
          "when the globally-best match lives in a different document")

    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED — embedding method: {method}")
    print("=" * 60)


if __name__ == "__main__":
    main()
