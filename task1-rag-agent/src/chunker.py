"""
chunker.py — Knowledge base ingestion and chunking.

Design rationale:
  - We keep chunks small (one Q&A pair or one issue block) so that each
    embedded vector represents a single, coherent concept.
  - Source metadata (doc name + chunk index) is stored alongside every
    chunk so the answer generation step can cite the right document.
  - No external vector DB required for prototype — flat JSON is sufficient
    for ~50 chunks at Clearpath's current KB scale.
"""

import re
from pathlib import Path
from typing import List, Dict


def chunk_faq(text: str, source: str) -> List[Dict]:
    """Split FAQ text into one chunk per Q&A pair."""
    chunks = []
    blocks = re.split(r'\n\s*\n(?=Q:)', text.strip())
    for i, block in enumerate(blocks):
        block = block.strip()
        if block:
            chunks.append({
                "source": source,
                "chunk_index": i,
                "chunk_type": "faq",
                "text": block,
            })
    return chunks


def chunk_releases(text: str, source: str) -> List[Dict]:
    """Split release notes into one chunk per version block."""
    chunks = []
    blocks = re.split(r'\n(?=v\d+\.\d+)', text.strip())
    for i, block in enumerate(blocks):
        block = block.strip()
        if block:
            chunks.append({
                "source": source,
                "chunk_index": i,
                "chunk_type": "release_note",
                "text": block,
            })
    return chunks


def chunk_runbook(text: str, source: str) -> List[Dict]:
    """Split runbook into one chunk per Issue block."""
    chunks = []
    blocks = re.split(r'\n(?=Issue:)', text.strip())
    for i, block in enumerate(blocks):
        block = block.strip()
        if block:
            chunks.append({
                "source": source,
                "chunk_index": i,
                "chunk_type": "runbook",
                "text": block,
            })
    return chunks


def ingest_knowledge_base(kb_dir: str) -> List[Dict]:
    """
    Read all three KB files, chunk them, and return a flat list of chunk dicts.
    Each chunk: {source, chunk_index, chunk_type, text}
    """
    kb_path = Path(kb_dir)
    all_chunks = []

    faq_text = (kb_path / "clearpath_faq.txt").read_text()
    all_chunks.extend(chunk_faq(faq_text, "clearpath_faq.txt"))

    releases_text = (kb_path / "clearpath_releases.txt").read_text()
    all_chunks.extend(chunk_releases(releases_text, "clearpath_releases.txt"))

    runbook_text = (kb_path / "clearpath_runbook.txt").read_text()
    all_chunks.extend(chunk_runbook(runbook_text, "clearpath_runbook.txt"))

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_knowledge_base("../knowledge_base")
    print(f"Total chunks: {len(chunks)}")
    for c in chunks:
        preview = c["text"][:80].replace("\n", " ")
        print(f"  [{c['source']} #{c['chunk_index']}] {preview}...")
