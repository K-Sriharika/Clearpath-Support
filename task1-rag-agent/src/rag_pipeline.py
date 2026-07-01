"""rag_pipeline.py — Top-level pipeline: query → retrieve → answer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chunker import ingest_knowledge_base
from embedder import build_vector_store, load_vector_store, retrieve
from agent import answer_query

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
TOP_K = 3

def get_store(rebuild: bool = False):
    if rebuild:
        chunks = ingest_knowledge_base(str(KB_DIR))
        return build_vector_store(chunks)
    try:
        return load_vector_store()
    except FileNotFoundError:
        print("[pipeline] No vector store found — building now...")
        chunks = ingest_knowledge_base(str(KB_DIR))
        return build_vector_store(chunks)

def run_query(query: str, store=None, verbose: bool = False) -> dict:
    if store is None:
        store = get_store()
    retrieved = retrieve(query, store, top_k=TOP_K)
    if verbose:
        print(f"\n[retrieval] Top {TOP_K} chunks for: '{query}'")
        for score, chunk in retrieved:
            print(f"  {score:.3f}  {chunk['source']} #{chunk['chunk_index']}: "
                  f"{chunk['text'][:60].replace(chr(10), ' ')}...")
    return answer_query(query, retrieved, embedding_method=store.get("embedding_method", "tfidf-fallback"))

def interactive_loop():
    import json
    print("\nClearpath Support Agent — RAG Prototype")
    print("Type your question and press Enter. Type 'quit' to exit.\n")
    store = get_store()
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        result = run_query(query, store, verbose=True)
        print(f"\n{'='*60}")
        print(f"Answer:\n{result['answer']}")
        if result['sources_used']:
            print(f"\nSources: {', '.join(result['sources_used'])}")
        if not result['confident']:
            print(f"[Fallback reason: {result.get('fallback_reason')}]")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    interactive_loop()
