"""
build_index.py — One-time (or re-run-on-change) script that chunks the KB,
embeds the chunks, and loads them into the Chroma collection.

Recreate the whole store from scratch with:
    python src/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunker import ingest_knowledge_base
from embedder import embed_texts
from vector_store import index_chunks

KB_DIR = Path(__file__).parent.parent / "knowledge_base"


def main():
    print("=" * 60)
    print("Clearpath Vector Store — Indexing")
    print("=" * 60)

    chunks = ingest_knowledge_base(str(KB_DIR))
    print(f"\nChunks extracted: {len(chunks)}")
    for c in chunks:
        preview = c["text"][:70].replace("\n", " ")
        print(f"  [{c['source']} #{c['chunk_index']}] {preview}...")

    texts = [c["text"] for c in chunks]
    vectors, method = embed_texts(texts)
    print(f"\nEmbedding method: {method}")
    print(f"Vector dimensionality: {len(vectors[0])}")

    collection = index_chunks(chunks, vectors, reset=True)
    print(f"\n✓ Indexed {collection.count()} chunks into Chroma collection "
          f"'{collection.name}'")
    print(f"  Persisted at: {Path(__file__).parent.parent / 'chroma_db'}")


if __name__ == "__main__":
    main()
