"""build_index.py — One-time script to chunk and embed the KB."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chunker import ingest_knowledge_base
from embedder import build_vector_store

KB_DIR = Path(__file__).parent.parent / "knowledge_base"

def main():
    print("=" * 60)
    print("Clearpath RAG — Knowledge Base Indexing")
    print("=" * 60)
    chunks = ingest_knowledge_base(str(KB_DIR))
    print(f"\nChunks extracted: {len(chunks)}")
    for c in chunks:
        preview = c["text"][:70].replace("\n", " ")
        print(f"  [{c['source']} #{c['chunk_index']}] {preview}...")
    store = build_vector_store(chunks)
    print(f"\n✓ Index built. Method: {store['embedding_method']}, Chunks: {len(store['chunks'])}")

if __name__ == "__main__":
    main()
