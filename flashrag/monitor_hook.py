"""rag-stack integration hook — no-op when standalone.

This module exposes a process-global monitor instance that the pipeline
call-site patches consult. When no monitor is registered (the default for
standalone FlashRAG usage), :func:`get_monitor` returns ``None`` and every
patched call site short-circuits — the original FlashRAG behavior.

When rag-stack's :class:`FlashRAGQualityEvaluator` is driving the run, it
calls :func:`set_monitor` with a :class:`rag_stack.flashrag_quality_evaluator.monitor.Monitor`
instance before ``pipeline.run(...)`` and resets it to ``None`` afterward.

The hook intentionally does NOT depend on rag-stack — the Monitor duck-types
the ``record_*`` API. Any external receiver that implements those methods
works.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

_monitor: Optional[Any] = None

# Thread-local stack of (query_ids, step_idx) tuples. Pipeline-level patches
# push a context before calling out to lower layers (retriever / reranker /
# vectordb); those layers read :func:`current_query_context` to attribute
# their captured calls to the right queries / step without changing FlashRAG
# function signatures.
_local = threading.local()


def set_monitor(monitor: Optional[Any]) -> None:
    """Install ``monitor`` as the active receiver (or clear it with ``None``).

    Idempotent. Safe to call from anywhere in the rag-stack code path.
    """
    global _monitor
    _monitor = monitor


def get_monitor() -> Optional[Any]:
    """Return the active monitor or ``None`` for standalone runs."""
    return _monitor


@contextmanager
def query_context(query_ids: List[str], step_idx: int):
    """Push ``(query_ids, step_idx)`` onto the thread-local context stack.

    Used by pipeline-level patches before delegating to retriever / reranker.
    The inner patches read :func:`current_query_context` to attribute their
    events without needing to change FlashRAG's call signatures.
    """
    if not hasattr(_local, "stack"):
        _local.stack = []
    _local.stack.append((list(query_ids), int(step_idx)))
    try:
        yield
    finally:
        _local.stack.pop()


def current_query_context() -> Optional[Tuple[List[str], int]]:
    """Return the active ``(query_ids, step_idx)`` or ``None``.

    Inner-layer patches use this to learn which queries / step they're
    operating on. When ``None``, the patch falls back to placeholder
    query_ids so events still record (helpful for debugging) but
    attribution will be incomplete.
    """
    if not hasattr(_local, "stack") or not _local.stack:
        return None
    return _local.stack[-1]


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Best-effort token count for one text string.

    Used by the pipeline-side patches to populate ``input_tokens`` /
    ``output_tokens`` on monitor events. We prefer a real tokenizer when
    the caller has one in scope; otherwise we fall back to whitespace-split
    which is good enough for downstream cost-model normalization.
    """
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())


def count_doc_tokens(docs: Any, tokenizer: Optional[Any] = None) -> int:
    """Best-effort token sum over a retrieved-docs list.

    Accepts FlashRAG's per-item retrieval_result shape — a list of dicts
    with a ``"contents"`` field. Anything that doesn't look like that
    contributes zero; the caller can stuff its own count via ``extras``.
    """
    if not docs:
        return 0
    total = 0
    for d in docs:
        contents = d.get("contents") if isinstance(d, dict) else None
        total += count_tokens(contents or "", tokenizer)
    return total


# ---------------------------------------------------------------------------
# Centralized record helpers — called from low-layer call sites (Generator
# subclasses, BaseTextRetriever.batch_search, BaseReranker.rerank).
#
# Attribution: each helper reads :func:`current_query_context` to figure out
# the query_ids + step_idx that the wrapping pipeline pushed for this call.
# When no context is set (e.g. someone calls FlashRAG directly outside a
# rag-stack pipeline) the events are still recorded with placeholder
# ``__unattributed_<i>__`` query_ids so DAG coverage is best-effort.
# ---------------------------------------------------------------------------


def _attribute(n: int) -> Tuple[List[str], int]:
    """Return ``(query_ids, step_idx)`` for a batch of size ``n``."""
    ctx = current_query_context()
    if ctx is not None:
        qids = [str(q) for q in ctx[0][:n]]
        # Pad with placeholders if the batch is somehow larger than the
        # pushed context (shouldn't happen for FlashRAG but be defensive).
        if len(qids) < n:
            qids = qids + [f"__unattributed_{i}__" for i in range(len(qids), n)]
        return qids, ctx[1]
    return [f"__unattributed_{i}__" for i in range(n)], 0


def record_generate_call(
    generator: Any,
    prompts: List,
    outputs: List,
    latency_ms: float,
    **extras: Any,
) -> None:
    """Record a batched generator.generate call to the active Monitor.

    Called from every concrete ``Generator.generate(...)`` implementation
    after its own work returns. No-op when no monitor is installed —
    standalone FlashRAG usage gets zero overhead beyond one ``is None``
    check.

    Token counts: if the caller passes ``precise_input_tokens`` /
    ``precise_output_tokens`` (lists matching the batch size), those win
    — useful for backends like vLLM / OpenAI that expose exact counts.
    Otherwise fall back to the generator's tokenizer.encode (HF), and
    finally to whitespace split.
    """
    mon = get_monitor()
    if mon is None:
        return
    qids, step = _attribute(len(prompts))
    tok = getattr(generator, "tokenizer", None)
    # FlashRAG generators normally pass list[str]; FiD-style code paths
    # may pass list[list[str]] — coerce defensively so token counting
    # never crashes the run.
    prompts_str = [p if isinstance(p, str) else " ".join(p) for p in prompts]
    outputs_str = [str(o) for o in outputs]

    precise_in = extras.pop("precise_input_tokens", None)
    precise_out = extras.pop("precise_output_tokens", None)
    if precise_in is not None and len(precise_in) == len(prompts_str):
        input_tokens = [int(n) for n in precise_in]
    else:
        input_tokens = [count_tokens(p, tok) for p in prompts_str]
    if precise_out is not None and len(precise_out) == len(outputs_str):
        output_tokens = [int(n) for n in precise_out]
    else:
        output_tokens = [count_tokens(o, tok) for o in outputs_str]

    mon.record_generate_batch(
        query_ids=qids,
        step_idx=step,
        model_id=getattr(generator, "model_name", None)
            or getattr(generator, "generator_model_name", None),
        prompts=prompts_str,
        outputs=outputs_str,
        input_token_counts=input_tokens,
        output_token_counts=output_tokens,
        latency_ms=float(latency_ms),
        extras=dict(extras),
    )


def record_retrieve_call(
    retriever: Any,
    queries: List,
    doc_lists: List,
    latency_ms: float,
    **extras: Any,
) -> None:
    """Record a batched retriever.batch_search call to the active Monitor.

    Called from :class:`BaseTextRetriever.batch_search` after its decorated
    body returns. Captures the COMPOSITE retrieve operation (encode +
    vectordb lookup + optional rerank); the inner ``vectordb`` event in
    ``_batch_search`` and the optional ``rerank`` event in
    ``BaseReranker.rerank`` are separate and live alongside this one.
    """
    mon = get_monitor()
    if mon is None:
        return
    qids, step = _attribute(len(queries))
    tok = getattr(retriever, "tokenizer", None)
    mon.record_retrieve_batch(
        query_ids=qids,
        step_idx=step,
        model_id=getattr(retriever, "retrieval_method", None),
        queries=list(queries),
        doc_lists=list(doc_lists),
        input_token_counts=[count_tokens(q, tok) for q in queries],
        output_token_counts=[count_doc_tokens(d, tok) for d in doc_lists],
        latency_ms=float(latency_ms),
        extras=dict(extras),
    )
