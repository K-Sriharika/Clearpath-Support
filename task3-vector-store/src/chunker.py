"""
chunker.py — Splits the three Clearpath knowledge base files into
semantic-unit chunks (one Q&A pair, one version block, one issue).

This mirrors the chunking strategy used in Task 1: arbitrary character-count
splitting would cut a Q&A pair or a runbook issue in half and destroy the
thing that makes it retrievable as a coherent answer. Splitting on the
document's own structural markers (blank-line-separated blocks) keeps each
chunk self-contained.
"""

import re
from pathlib import Path
from typing import List, Dict


def chunk_faq(text: str, source: str) -> List[Dict]:
    """One chunk per Q/A pair."""
    blocks = re.split(r"\n(?=Q:)", text.strip())
    chunks = []
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
    """One chunk per version block (all bullets under one version header)."""
    blocks = re.split(r"\n(?=v\d)", text.strip())
    chunks = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if block:
            chunks.append({
                "source": source,
                "chunk_index": i,
                "chunk_type": "release_notes",
                "text": block,
            })
    return chunks


def chunk_runbook(text: str, source: str) -> List[Dict]:
    """One chunk per Issue/Steps pair."""
    blocks = re.split(r"\n(?=Issue:)", text.strip())
    chunks = []
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
    Read all three KB files, chunk them, and return a flat list of chunk
    dicts: {source, chunk_index, chunk_type, text}
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
