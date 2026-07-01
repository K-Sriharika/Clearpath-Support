"""
agent.py — RAG answer generation with fallback handling.

Design rationale:

MODEL TIER:
  We use claude-haiku-3-5 for answer generation.  It is the cheapest
  Anthropic model that handles instruction-following reliably.  At
  Clearpath's 300 tickets/day, assuming avg 400 prompt tokens + 150
  output tokens per query:
    300 * 550 tokens = 165k tokens/day
  claude-haiku-3-5 costs ~$0.25/M input + $1.25/M output, so:
    ~$0.04/day for answer generation.
  Haiku is more than capable for retrieval-grounded Q&A over short passages.
  We reserve Sonnet for the judge evaluation (higher reasoning needed).

PROMPT TOKEN BUDGET:
  System prompt: ~200 tokens (fixed, cached by Anthropic's prompt caching).
  Retrieved context: top-3 chunks, avg ~80 tokens each = ~240 tokens.
  User question: ~20 tokens.
  Total prompt: ~460 tokens — well within Haiku's context window.
  We intentionally limit retrieved chunks to 3 to cap the prompt size.

FALLBACK STRATEGY:
  Low-confidence retrieval is detected two ways:
    1. Score threshold: if the top cosine score < CONFIDENCE_THRESHOLD_TFIDF (or _VOYAGE),
       no chunk is reliably relevant.
    2. Model self-assessment: the system prompt instructs the model to
       return a sentinel JSON field "confident": false when it cannot
       ground its answer in the provided context.
  Either condition triggers the fallback response — a polite message
  that acknowledges the limit and offers escalation.  This directly
  addresses the Head of Support Operations' callout about hallucinated
  answers when the KB is out of date.

GROUNDING:
  The system prompt explicitly forbids drawing on outside knowledge and
  requires the model to cite which chunk(s) it used.  Chunks are labeled
  [SOURCE #N] in the prompt so the model can reference them precisely.
"""

import os
import json
import urllib.request
from typing import List, Tuple, Dict, Optional

# Cosine score below this → low-confidence retrieval fallback.
# TF-IDF scores are lower than semantic embedding scores (smaller vocabulary
# overlap vs dense semantic space), so we use a lower threshold for TF-IDF mode.
# With Voyage embeddings, in-scope queries typically score 0.4-0.8; threshold 0.20.
# With TF-IDF, in-scope queries score 0.12-0.5; threshold 0.12.
CONFIDENCE_THRESHOLD_TFIDF = 0.12
CONFIDENCE_THRESHOLD_VOYAGE = 0.20

# Anthropic API settings
# Using Haiku for answer generation (cost-optimised)
ANSWER_MODEL = "claude-haiku-4-5"
# Using Sonnet for judge evaluation (higher reasoning)
JUDGE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_VERSION = "2023-06-01"


def _call_anthropic(
    messages: List[Dict],
    system: str,
    model: str,
    max_tokens: int = 512,
) -> str:
    """
    Minimal HTTP wrapper around /v1/messages.
    Returns the text content of the first content block.

    We avoid the full anthropic SDK to keep dependencies minimal —
    this prototype only needs a single endpoint.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    # Extract text from the first content block
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


# ── System prompts ────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are a support agent assistant for Clearpath Support Systems.
Your sole job is to answer support questions using ONLY the knowledge base excerpts provided.

Rules:
1. Answer ONLY from the provided context chunks. Do not use any outside knowledge.
2. If the context does not contain enough information to answer confidently, set "confident" to false.
3. Cite which chunk(s) you used by referencing their [SOURCE] label.
4. Be concise. Support agents need fast, actionable answers.
5. If steps are involved, preserve them as a numbered list.
6. Never invent information not present in the chunks.

Respond in this exact JSON format:
{
  "confident": true,
  "answer": "Your answer here, citing [SOURCE labels] where relevant.",
  "sources_used": ["source name 1", "source name 2"]
}

If you cannot answer from the provided context:
{
  "confident": false,
  "answer": "",
  "sources_used": []
}"""


FALLBACK_MESSAGE = (
    "I wasn't able to find a confident answer to that question in the Clearpath "
    "knowledge base. This might mean the topic isn't covered yet or the question "
    "is phrased in a way that doesn't match existing documentation.\n\n"
    "**Suggested next steps:**\n"
    "- Try rephrasing the question with different keywords.\n"
    "- Escalate to your team lead or the engineering channel.\n"
    "- If this is a recurring gap, flag it so the knowledge base can be updated."
)


# ── Core answer function ──────────────────────────────────────────────────────

def answer_query(
    query: str,
    retrieved: List[Tuple[float, Dict]],
    embedding_method: str = "tfidf-fallback",
) -> Dict:
    """
    Given a query and retrieved (score, chunk) pairs, generate a grounded answer.

    Returns:
        {
            "query": str,
            "answer": str,
            "confident": bool,
            "sources_used": List[str],
            "top_score": float,
            "fallback_reason": Optional[str],
        }
    """
    top_score = retrieved[0][0] if retrieved else 0.0

    # Select threshold based on embedding method.
    # TF-IDF operates in a sparser vector space so in-scope scores are lower.
    threshold = (CONFIDENCE_THRESHOLD_TFIDF
                 if embedding_method == "tfidf-fallback"
                 else CONFIDENCE_THRESHOLD_VOYAGE)

    # ── Fallback: retrieval confidence too low ────────────────────────────────
    if top_score < threshold:
        return {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "confident": False,
            "sources_used": [],
            "top_score": top_score,
            "fallback_reason": f"Top retrieval score {top_score:.3f} below {embedding_method} threshold {threshold}",
        }

    # ── Build context block from retrieved chunks ─────────────────────────────
    context_parts = []
    for i, (score, chunk) in enumerate(retrieved):
        label = f"{chunk['source']} (chunk {chunk['chunk_index']})"
        context_parts.append(f"[SOURCE {i+1}: {label}]\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Knowledge base excerpts:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    # ── Call the LLM ──────────────────────────────────────────────────────────
    raw = _call_anthropic(
        messages=[{"role": "user", "content": user_message}],
        system=ANSWER_SYSTEM_PROMPT,
        model=ANSWER_MODEL,
        max_tokens=512,
    )

    # ── Parse JSON response ───────────────────────────────────────────────────
    try:
        # Strip possible markdown code fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # Model returned plain text — treat as answer, mark confident=True
        parsed = {"confident": True, "answer": raw, "sources_used": []}

    confident = parsed.get("confident", True)
    answer_text = parsed.get("answer", "").strip()
    sources_used = parsed.get("sources_used", [])

    # ── Model self-assessed as not confident ──────────────────────────────────
    if not confident or not answer_text:
        return {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "confident": False,
            "sources_used": [],
            "top_score": top_score,
            "fallback_reason": "Model self-assessed as not confident given retrieved context",
        }

    return {
        "query": query,
        "answer": answer_text,
        "confident": True,
        "sources_used": sources_used,
        "top_score": top_score,
        "fallback_reason": None,
    }


# ── Judge evaluation ──────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing a support agent AI's answers.
You will be given:
- A question
- The knowledge base excerpts that were retrieved
- The AI's answer

Score the answer on two dimensions, each 1-5:

GROUNDEDNESS (1-5): Does every claim in the answer appear in the retrieved excerpts?
  5 = Fully grounded; every claim traceable to the excerpts
  4 = Mostly grounded; one minor detail not clearly sourced
  3 = Partially grounded; some claims not in excerpts
  2 = Mostly ungrounded; most claims unsupported
  1 = Hallucinated; answer contradicts or ignores the excerpts

CORRECTNESS (1-5): Is the answer factually correct and complete relative to the knowledge base?
  5 = Correct and complete; all key information included
  4 = Correct but slightly incomplete
  3 = Partially correct; missing or slightly wrong on some points
  2 = Mostly incorrect
  1 = Wrong or dangerously misleading

Also note:
- PASS/FAIL for fallback handling: if the question is out-of-scope, did the agent correctly decline?

Respond in this exact JSON format:
{
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "fallback_correct": true/false/null,
  "reasoning": "Brief explanation of scores (2-3 sentences).",
  "flag": null
}

Set "flag" to a short string if the answer contains dangerous misinformation; null otherwise."""


def judge_answer(
    query: str,
    retrieved_context: str,
    agent_answer: str,
    is_out_of_scope: bool = False,
) -> Dict:
    """
    Run LLM-as-a-judge evaluation on one answer.
    Uses Sonnet (higher reasoning tier) for reliable rubric application.

    Cost note: Judge calls are only made during evaluation runs, not
    in production.  5-10 judge calls during eval < $0.01 total.
    """
    user_content = (
        f"Question: {query}\n\n"
        f"Retrieved knowledge base excerpts:\n{retrieved_context}\n\n"
        f"AI's answer: {agent_answer}\n\n"
        f"Is this question intentionally out-of-scope? {is_out_of_scope}"
    )

    raw = _call_anthropic(
        messages=[{"role": "user", "content": user_content}],
        system=JUDGE_SYSTEM_PROMPT,
        model=JUDGE_MODEL,
        max_tokens=400,
    )

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}
