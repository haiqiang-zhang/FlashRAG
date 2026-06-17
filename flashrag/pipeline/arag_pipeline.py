"""A-RAG: Agentic RAG with hierarchical retrieval interfaces (arXiv:2602.03442).

Implemented IN the FlashRAG backend (per rag-stack's "all RAG evaluation lives
in the backend" philosophy), alongside the other vendored pipelines.

A-RAG exposes several retrieval *tools* directly to the LLM and lets it choose,
adaptively, which one to call at each step of a Thought / Action / Observation
loop (a ReAct-style agent generalised from one retriever to several retrieval
granularities):

  * ``Search[query]``  — **semantic** (dense) retrieval over the corpus.
  * ``Keyword[query]`` — **keyword** (BM25, exact-term) retrieval; best for
                         names / numbers / codes / exact phrases. Available only
                         when a BM25 index is wired (``config["bm25_index_path"]``);
                         absent → the tool gracefully reports it is unavailable.
  * ``Read[doc_id]``   — **chunk read**: fetch the full text of a specific chunk
                         by id (the ids are surfaced in search observations), so
                         the agent can drill into a promising hit.
  * ``Finish[answer]`` — terminate with the final answer.

Every generator / retriever call site is wrapped in ``query_context`` so the
rag-stack monitor attributes component calls to per-query ``trace_v1`` payloads
(``search``/``batch_search`` are already ``@monitor_retrieve``-decorated in the
fork; the recorded ``model_id`` is the retriever's ``retrieval_method`` — ``e5``
for dense, ``bm25`` for keyword — which the trace-driven cost model uses to price
the two retrieval kinds differently). ``Read`` is a local corpus lookup — no
model compute — so it is intentionally NOT a traced node (cost 0).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional

from flashrag.monitor_hook import query_context
from flashrag.pipeline import BasicPipeline
from flashrag.prompt import PromptTemplate
from flashrag.retriever.utils import load_corpus
from flashrag.utils import get_generator, get_retriever

logger = logging.getLogger("RAG-Stack")


class ARAGPipeline(BasicPipeline):
    """A-RAG agent loop with semantic-search + keyword-search + chunk-read tools."""

    system_prompt = ""
    # One-shot exemplar: small instruct models do not follow the tool format
    # zero-shot. The example shows Search -> Read -> Finish so the model learns
    # to (a) search, (b) drill into a returned id, (c) conclude.
    user_prompt = (
        "Answer the question using a knowledge base you can query with tools.\n"
        "At each step write a Thought, then exactly one Action. Actions:\n"
        "(1) Search[query] — semantic search; returns passages each tagged with "
        "an id like [id=12].\n"
        "(2) Keyword[query] — keyword (exact-term) search; best for names, "
        "numbers, codes, or exact phrases. Also returns id-tagged passages.\n"
        "(3) Read[id] — read the full passage with that id (use ids from a "
        "search result to get more detail).\n"
        "(4) Finish[answer] — give the final answer.\n"
        "After Search/Keyword/Read the system returns an Observation. Always "
        "search before answering; Finish as soon as you can answer.\n\n"
        "Example:\n"
        "Question: Where was the author of \"Walden\" born?\n"
        "Thought 1: I need the author of \"Walden\".\n"
        "Action 1: Search[author of Walden]\n"
        "Observation 1: [id=7] Walden is an 1854 book by Henry David Thoreau...\n"
        "Thought 2: The author is Thoreau; I need his birthplace — let me read "
        "that passage.\n"
        "Action 2: Read[7]\n"
        "Observation 2: Henry David Thoreau was born on July 12, 1817, in "
        "Concord, Massachusetts...\n"
        "Thought 3: He was born in Concord, Massachusetts.\n"
        "Action 3: Finish[Concord, Massachusetts]\n\n"
        "Now solve the real question the same way.\n"
        "Question: {question}\n"
        "Thought 1:"
    )

    _SEARCH_RE = re.compile(r"Search\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)
    _KEYWORD_RE = re.compile(r"Keyword\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)
    _READ_RE = re.compile(r"Read\s*\[\s*([^\]]+?)\s*\]", re.DOTALL | re.IGNORECASE)
    _FINISH_RE = re.compile(r"Finish\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)

    # Observation truncation — corpora with large chunks (dragonball ~1.7k
    # tokens/chunk) would blow the context window otherwise. Read gets a larger
    # budget than search (the point of Read is to see more of one chunk).
    _SEARCH_CHARS_PER_DOC = 240
    _READ_CHARS = 1200

    def __init__(
        self,
        config,
        prompt_template=None,
        max_iter=8,
        retriever=None,
        generator=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
        super().__init__(config, prompt_template)
        self.generator = generator if generator is not None else get_generator(config)
        self.retriever = retriever if retriever is not None else get_retriever(config)
        self.bm25_retriever = self._maybe_build_bm25(config)
        self.max_iter = int(max_iter)
        self.stop_tokens = ["Observation", "<|im_end|>", "<|endoftext|>"]

        # Chunk-read index: id -> full contents. Built once from the same corpus
        # the dense index was built over.
        corpus = load_corpus(config["corpus_path"])
        self.id2contents: Dict[str, str] = {
            str(row["id"]): str(row.get("contents", "")) for row in corpus
        }

    @staticmethod
    def _maybe_build_bm25(config):
        """Build a BM25 keyword retriever from ``config['bm25_index_path']``.

        Returns ``None`` (Keyword tool disabled) when no BM25 index is wired —
        so the pipeline degrades cleanly to semantic + chunk-read.
        """
        def g(key, default=None):
            try:
                return config[key]
            except Exception:
                return default

        bm25_path = g("bm25_index_path")
        if not bm25_path:
            return None
        bm25_cfg = {
            "retrieval_method": "bm25",
            "index_path": bm25_path,
            "corpus_path": g("corpus_path"),
            "retrieval_topk": g("retrieval_topk", 4),
            "bm25_backend": g("bm25_backend", "bm25s"),
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": g("silent_retrieval", True),
            "save_dir": g("save_dir", "."),
        }
        try:
            from flashrag.retriever import BM25Retriever
            return BM25Retriever(bm25_cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[a_rag] BM25 retriever unavailable ({e}); Keyword tool disabled."
            )
            return None

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    @classmethod
    def _doc_title(cls, contents: str) -> str:
        return contents.split("\n")[0][:120]

    def _search_observation(self, docs: List[Dict]) -> str:
        parts = []
        for doc in docs:
            doc_id = str(doc.get("id", "?"))
            contents = str(doc.get("contents", ""))
            title = self._doc_title(contents)
            body = " ".join(contents.split("\n")[1:]) or title
            if len(body) > self._SEARCH_CHARS_PER_DOC:
                body = body[: self._SEARCH_CHARS_PER_DOC] + "…"
            parts.append(f"[id={doc_id}] {title}: {body}")
        return " ".join(parts) if parts else "No results found."

    def _read_observation(self, doc_id: str) -> str:
        contents = self.id2contents.get(doc_id)
        if contents is None:
            return f"No passage with id {doc_id}."
        return contents[: self._READ_CHARS] + ("…" if len(contents) > self._READ_CHARS else "")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", prompts)
        dataset.update_output("finish_flag", [False] * len(prompts))
        dataset.update_output("arag_round", [1] * len(prompts))
        dataset.update_output("retrieval_results", [{} for _ in range(len(prompts))])
        dataset.update_output("retrieved_times", [0] * len(prompts))

        # Tool-usage tally — Read/keyword aren't otherwise observable (Read is a
        # local lookup, not a traced node), so log which tools the agent actually
        # chose this run. Makes "did the model use Read/Keyword?" answerable.
        tool_counts: Counter = Counter()

        for step_idx in range(self.max_iter + 1):
            frontier = [item for item in dataset if not item.finish_flag]
            if not frontier:
                break
            if step_idx == self.max_iter:
                for item in frontier:
                    item.pred = "No valid answer found"
                    item.finish_flag = True
                    item.finish_reason = "Reach max iterations"
                break

            with query_context(
                [str(item.id) for item in frontier], step_idx=step_idx
            ):
                outputs = self.generator.generate(
                    [
                        self.prompt_template.truncate_prompt(item.prompt)
                        for item in frontier
                    ],
                    stop=self.stop_tokens,
                )
            if step_idx == 0 and outputs:
                logger.info(f"[a_rag] round-0 sample output: {outputs[0][:400]!r}")

            dense_searches = []    # [{'item':, 'query':}]
            bm25_searches = []     # [{'item':, 'query':}]
            for item, out in zip(frontier, outputs):
                out = out.strip()
                item.prompt = item.prompt + " " + out
                finish_m = self._FINISH_RE.findall(out)
                search_m = self._SEARCH_RE.findall(out)
                keyword_m = self._KEYWORD_RE.findall(out)
                read_m = self._READ_RE.findall(out)
                n = item.arag_round
                if finish_m:
                    tool_counts["Finish"] += 1
                    item.pred = finish_m[-1].strip()
                    item.finish_flag = True
                    item.finish_reason = "Finished"
                elif search_m:
                    tool_counts["Search"] += 1
                    dense_searches.append({"item": item, "query": search_m[-1].strip()})
                elif keyword_m:
                    tool_counts["Keyword"] += 1
                    if self.bm25_retriever is not None:
                        bm25_searches.append({"item": item, "query": keyword_m[-1].strip()})
                    else:
                        item.prompt += (
                            f"\nObservation {n}: Keyword search is unavailable; "
                            f"use Search instead.\nThought {n + 1}:"
                        )
                        item.arag_round = n + 1
                elif read_m:
                    tool_counts["Read"] += 1
                    doc_id = read_m[-1].strip()
                    item.prompt += (
                        f"\nObservation {n}: {self._read_observation(doc_id)}"
                        f"\nThought {n + 1}:"
                    )
                    item.arag_round = n + 1
                else:
                    tool_counts["none"] += 1
                    item.pred = out
                    item.finish_flag = True
                    item.finish_reason = "Normal finish without action"

            self._run_search_batch(dense_searches, self.retriever, step_idx)
            self._run_search_batch(bm25_searches, self.bm25_retriever, step_idx)

        logger.info(f"[a_rag] tool usage this run: {dict(tool_counts)}")
        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )
        return dataset

    def _run_search_batch(self, batch: List[Dict], retriever, step_idx: int) -> None:
        """Run one batched retrieval (dense or bm25) and fold results back as
        Observations. ``retriever`` is the model-compute tool; its call is
        wrapped in ``query_context`` so the monitor records it (with the
        retriever's ``retrieval_method`` as the trace ``model_id``)."""
        if not batch:
            return
        with query_context(
            [str(s["item"].id) for s in batch], step_idx=step_idx
        ):
            docs_per_query = retriever.batch_search([s["query"] for s in batch])
        for s, docs in zip(batch, docs_per_query):
            item = s["item"]
            item.retrieval_results[item.retrieved_times] = {
                "query": s["query"],
                "docs": list(docs),
            }
            item.retrieved_times += 1
            n = item.arag_round
            item.prompt += (
                f"\nObservation {n}: {self._search_observation(docs)}"
                f"\nThought {n + 1}:"
            )
            item.arag_round = n + 1
