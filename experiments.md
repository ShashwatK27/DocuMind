# DocuMind — Experiment Log

A living record of what we changed, what we measured, and what each result led us
to decide. Companion to [`decision.md`](decision.md) (which records *why* we chose
what we chose). Raw outputs live in [`experiment/`](experiment/); the driver is
[`notebooks/rag_1.ipynb`](notebooks/rag_1.ipynb).

> **How to read the metrics.** The golden set was **expanded from 10 → 43
> questions (39 answerable, 4 unanswerable) on 2026-10-10**, matching the roadmap's
> call for a 30–50 question set. Sections **E1–E6 were measured on the original
> 10-question set** (steps of ~1/9); **E7 onward and the selected config are
> (re-)confirmed on the 43-question set** (steps of ~1/39 — far more reliable).
> **Page Recall@5** = "did any of the top-5 final chunks come from an expected
> page?" — a page-level diagnostic, not proof the answer text was present.
> **Content coverage** (E7+) = fraction of a question's answer terms present in the
> delivered context. **MRR** is on the final, post-diversification top-5. Latencies
> are wall-clock on CPU.

Fixed harness (held constant unless named as the variable): the same 10-question
golden set, the same `page_recall_at_5` / MRR definitions, an isolated in-memory
Qdrant index per ingestion configuration, and cosine distance.

---

## Timeline of changes

### 2026-10-15 — Ragas: decoupled via a JSONL seam (local, not CI)

Wired Ragas as a **local, opt-in** evaluation rather than a CI gate. Ragas can't
share an interpreter with the validated langchain stack — its current release
(`0.4.3`) fails to import even in a clean venv (`ModuleNotFoundError:
langchain_community.chat_models.vertexai`), and unpinning to fix it cascades into
`langchain-core`/`langgraph`/`numpy`/`pyarrow` conflicts; it also pulls a
hosted-LLM stack against local-first. So the two stacks are decoupled by a JSONL
hand-off: `python -m eval --dump-ragas experiment/ragas_input.jsonl` (service env)
emits `{user_input, retrieved_contexts, response, reference, reference_contexts}`
per answerable question, and `scripts/ragas_eval.py` scores it in a dedicated
Ragas venv with **Ollama as judge + MiniLM embeddings** (faithfulness, answer
relevancy, context precision/recall). The producer is shipped and tested; the
scorer runs once Ragas is installed in its own env. CI keeps the deterministic
retrieval gate (D-18); the LLM-judged Ragas metrics are nondeterministic and stay
a local report. D-19's answer-accuracy + 100% refusal remain the dependency-free
quality proxy in the meantime. Recorded as D-21.

### 2026-10-15 — Generation-aware eval: answer-term accuracy vs context coverage

Added an opt-in `python -m eval --generate` mode that runs the full pipeline
through the LLM and scores each answer for **answer-term accuracy** — the share
of ground-truth terms the *generated answer* actually states — alongside the
existing **context coverage** (terms present in the *retrieved context*). Coverage
asks "did we retrieve it?"; accuracy asks "did we say it?", so the gap isolates
generation loss. Measured on the 4-doc corpus with `llama3.2:3b` (temp 0):

| metric | value | reads as |
|---|---:|---|
| context coverage | 0.754 | the answer was in the retrieved context |
| **answer accuracy** | **0.500** | the generated answer stated the ground-truth terms |
| accuracy ≥ 0.5 | 0.525 | share of answers that mostly stated it |
| unanswerable refused | **1.000** | all 4 unanswerable questions correctly refused |

The **~25-point coverage→accuracy gap** is the headline: retrieval is strong, but
the 3B model frequently has the right context and still omits the exact terms.
The `--generate` report lists the "generation loss" cases (coverage ≥ 0.5 but
accuracy < 0.5) to triage. **Caveat (honest):** accuracy is exact-term recall
with the same ≥5-char tokenizer as coverage, so it undercounts correct answers
that use a synonym or a different inflection (e.g. the model's singular "project"
vs the ground truth's "projects"). It is a *proxy that flags candidates for
review*, not a verdict — the gap and the per-question list are the signal. This
is the natural successor the earlier rounds kept flagging (open issue #2/#6:
measure accuracy, not just coverage, to catch the wrong-list case). Retrieval
metrics are unchanged (deterministic); generation is not scored in CI (no Ollama
there). Recorded as D-19.

### 2026-10-14 — Query-aware fusion: closed 2 of the 3 residual misses

Hybrid augmentation (above) put the exact-match chunks *in the candidate pool*,
but a diagnostic showed the cross-encoder still ranked two of them at ~rank 7 —
just outside the top-5 after page diversification. The correct chunks were
present and strongly favoured by BM25; only the final ordering was wrong. So
instead of a bigger `reranker_top_k` (which would game Recall@5), I added a
**final-stage query-aware fusion**: standardize the reranker and BM25 scores per
query and rank on `reranker_z + α · bm25_z` (D-17). Swept α on the 4-doc corpus:

| α | Recall@5 | MRR | coverage | cov ≥ 0.5 | misses |
|---:|---:|---:|---:|---:|---:|
| 0.00 (pure reranker) | 0.949 | 0.890 | 0.744 | 0.763 | 3 |
| 0.10 | 0.966 | 0.896 | 0.744 | 0.763 | 2 |
| **0.15 (adopted)** | **0.983** | **0.897** | **0.754** | **0.780** | **1** |
| 0.30 | 0.983 | 0.889 | 0.753 | 0.780 | 1 |
| 0.50 | 0.983 | 0.857 | 0.767 | 0.797 | 1 |
| 1.00 | 0.949 | 0.781 | 0.754 | 0.780 | 3 |

**α = 0.15 is the knee.** It recovers both closeable misses (the Transformer
*Adam optimizer* page and the RAG *FAISS/HNSW* page), lifts MRR *above* the
pure-reranker baseline (0.890 → 0.897), raises coverage, and introduces **no new
misses**. The curve also shows the failure mode of over-weighting lexical signal:
past α ≈ 0.5 MRR erodes, and at α = 1.0 two *new* exact-match questions (BERT
acronym, LoRA/QLoRA) get displaced — recall falls back to 0.949. A light touch
wins. **The one surviving miss is deliberate:** the roadmap *"Which Ragas metrics
… for the golden set?"* question. The full four-metric list is enumerated only on
page 1, but the query says "golden set" and page 2 *is* the golden-set paragraph
(and genuinely names Ragas + faithfulness + context precision). The reranker
confidently returns page 2 — a defensible answer — so relabelling page 1 → 2 to
turn the metric green would be gaming, not a fix. Left standing as real signal.

### 2026-10-14 — Hybrid retrieval: measured, tuned, adopted

Ran the hybrid-search experiment that D-15 set up, on the 4-document corpus.

| metric | dense | naive RRF | **augmented (shipped)** |
|---|---:|---:|---:|
| Page Recall@5 | 0.932 | 0.932 | **0.949** |
| MRR | 0.852 | 0.845 | **0.890** |
| Content coverage | 0.712 | 0.681 | **0.744** |
| coverage ≥ 0.5 | 0.763 | 0.678 | **0.763** |

Two hybrid designs, one clear winner. **Naive Reciprocal-Rank Fusion** was a wash:
it recovered the two predicted exact-match misses but, because RRF re-truncates to
k, evicted two previously-correct dense chunks from the candidate pool before the
reranker saw them — net zero on recall, worse on coverage. **Candidate
augmentation** (dense top-k ∪ BM25 top-k, reranked) keeps every dense chunk and
only adds BM25's lexical hits, so the cross-encoder can only benefit: it beats
dense on every axis with no regressions and is now the default (D-16). Verified
end-to-end in the service, including BM25 rebuilt from a Qdrant scroll after a
simulated restart. Three misses remain (RAG FAISS/HNSW page, Transformer
optimizer page, roadmap "Ragas") — the next lever.

### 2026-10-13 — Shipped the service and a multi-document corpus

Two steps, both toward turning the locked pipeline into a portfolio-grade system
(see `decision.md` D-14, D-15).

**Served the pipeline.** Filled `app/` with a FastAPI service that encodes the
selected configuration, with `/ingest`, `/query`, `/feedback`, `/health` and a
demo UI, plus Docker/compose and a README that leads with the results. Added a
committed, source-aware evaluation harness (`eval/`, `python -m eval`) that runs
the *real* `app` pipeline so the numbers can never drift from the service; it
reproduced the notebook's n=43 figures exactly (Recall@5 1.000, MRR 0.944).

**Added a 4-document corpus.** Fetched three public arXiv papers (Transformer,
BERT, RAG) via `scripts/fetch_corpus.py` and hand-grounded 20 new questions
(golden set now 63; 59 answerable). The index grew 73 → 569 chunks. Results on
the multi-document corpus:

| metric | single-doc (n=39) | 4-doc (n=59) |
|---|---:|---:|
| Page Recall@5 | 1.000 | **0.932** |
| MRR | 0.944 | **0.852** |
| Content coverage | 0.759 | **0.712** |
| coverage ≥ 0.5 | 0.795 | **0.763** |

The drop is the point: the saturated single-doc metric hid everything. All **4
genuine Recall@5 misses are exact-match cases** — the queries hinge on rare
tokens (*Ragas*, *Adam*, *FAISS*, *HNSW*) that dense retrieval dilutes because a
semantically-similar document outranks the correct page. Two are cross-document
(roadmap questions swamped by the RAG paper); two are within-document (the one
page carrying the exact term is outranked by prose about the same topic). This is
the first concrete evidence that **hybrid search** — rejected on the single doc
(D-11) — should help, and it is the next experiment to run (D-15).

### 2026-10-10 — Golden set expanded 10 → 43 questions

Grounded every question in the roadmap PDF across all 10 pages; 39 answerable + 4
unanswerable. A sanity check (answer terms must appear on the claimed page) caught
and fixed **2 wording bugs** (acronyms HNSW/IVF are too short to match; "leaking"
vs the doc's "leak"). Re-ran the feature-engineering suite at n=39: the selection
(`enriched-500/150`) **held** with real signal, and a coverage-vs-recall trade-off
with `enriched-600/150` emerged that the tiny set had hidden (see E7, D-13).

### 2026-10-10 — Notebook review fixes applied

Applied before any new experiment so later runs are trustworthy. See `decision.md`
D-1…D-5 for rationale.

| # | Change | Cell | Effect |
|---|--------|------|--------|
| 1 | Ollama **connection probe** before enabling generation | §9 setup | Missing/stopped Ollama now degrades to retrieval-only instead of crashing the eval loops |
| 2 | Chunk-count table uses the **same separators + (size, overlap) pairs** as the real experiment | §4 | Descriptive counts now match what is actually evaluated |
| 3 | **Data-driven chunk selection** (`SELECTED_CHUNK_SIZE/OVERLAP`) replaces hard-coded `600/100` downstream | §15 → §17/18/19 | The chunk choice is now derived from measured MRR, not asserted |
| 4 | `RESET_QDRANT` **rmtree retry** on Windows file locks | §6 | Reset no longer intermittently fails with `PermissionError` on win32 |
| 5 | Cross-Encoder **warm-up** call after load | §10 | First *measured* rerank no longer absorbs one-time init cost → cleaner latency sweeps |

Tooling: notebook executed headlessly with `nbclient` against the project `venv`
(Python 3.11.9) and local Ollama `llama3.2:3b`.

---

## Experiments & results

All numbers below are from [`experiment/`](experiment/) (prior committed runs).
Recall/MRR are deterministic for a fixed config; only latency varies run-to-run.

### E1 — Chunk size & proportional overlap  *(numbers now n=39)*
Variable: `chunk_size` / `chunk_overlap`. Held: top-k=10, reranker on, reranker_top_k=5, max_per_page=2, MiniLM-L6-v2.

| config | #chunks | Page Recall@5 | MRR |
|--------|--------:|--------------:|----:|
| 400 / 75   | 77 | 0.974 | 0.874 |
| 600 / 100 | 50 | 0.974 | 0.850 |
| 800 / 150  | 41 | 0.949 | 0.893 |
| 1000 / 150 | 32 | 0.923 | 0.833 |
| 1200 / 200 | 28 | 0.949 | 0.796 |

**Finding.** On the bigger set the plain-chunk MRR is close across 400–800 (0.85–0.89)
and degrades for the largest chunks (passage diluted). This plain sweep is only an
initial screen — the **chunk choice is ultimately made jointly with overlap and
enrichment in E7**, which is why the early 600/100 pick was revisited there.

### E2 — Retrieval top-k sweep
Variable: `retrieval_top_k`. Held: 600/100, reranker on, reranker_top_k=5, max_per_page=2.

*(numbers now n=39)*

| top-k | Page Recall@5 | MRR | rerank ms |
|------:|--------------:|----:|----------:|
| 3  | 0.821 | 0.756 | 91 |
| 5  | 0.923 | 0.852 | 139 |
| 8  | 0.897 | 0.834 | 242 |
| 10 | 0.949 | 0.893 | 309 |
| 15 | **1.000** | **0.925** | 441 |
| 20 | 0.974 | 0.915 | 604 |

**Finding.** Recall and MRR climb with top-k and plateau around **15–20** (n=9 had
made top-k=20 look uniquely special — a small-sample artifact). On the plain
600/100 setup top-k=15 even edges top-k=20. The validated E7 pipeline keeps
top-k=20 (where enrichment restores recall to 1.0), but **top-k=15 is now a cheaper
candidate worth A/B-ing**. → D-4.

### E3 — Reranker ON vs OFF
Variable: `reranker_enabled` × top-k ∈ {3,10,20}. Held: 600/100, reranker_top_k=5, max_per_page=2.
`candidate_page_recall` is measured on the raw candidates (a reranker cannot
recover a page retrieval never returned).

*(numbers now n=39)*

| top-k | mode | candidate recall | Page Recall@5 | MRR | total ms |
|------:|------|-----------------:|--------------:|----:|---------:|
| 3  | off | 0.821 | 0.821 | 0.705 | 15 |
| 3  | on  | 0.821 | 0.821 | 0.756 | 101 |
| 10 | off | 0.949 | 0.923 | 0.729 | 15 |
| 10 | on  | 0.949 | 0.949 | **0.893** | 314 |
| 20 | off | **1.000** | 0.923 | 0.729 | 15 |
| 20 | **on** | **1.000** | **0.974** | **0.915** | 607 |

**Finding.** The bigger set gives the reranker **unambiguous, broad value**: it
lifts MRR at every top-k (e.g. 0.729→0.893 at k=10, 0.729→0.915 at k=20) and recall
at k=20 (0.923→0.974). On n=9 the effect had only shown at top-k=20 — a small-sample
artifact. The reranker cannot recover a page absent from candidates (candidate
recall caps final recall), so it still pairs with a wide pool.
**Decision:** reranker **ON**. → D-4.

### E4 — Page diversification (`max_chunks_per_page`)  *(n=9; orphan CSV, not refreshed)*
Variable: `max_chunks_per_page`. Held: 600/100, top-k=20, reranker on, reranker_top_k=5.

| max/page | Page Recall@5 | MRR |
|----------|--------------:|----:|
| unlimited | 1.000 | 0.806 |
| **1** | **1.000** | **0.815** |
| 2 | 1.000 | 0.806 |
| 3 | 1.000 | 0.806 |

**Finding.** Capping at **1 chunk per page** slightly improves MRR (0.806→0.815)
with no recall loss — forcing page diversity surfaces a relevant page a notch
higher. Small effect (one question), but free.
**Decision:** `max_chunks_per_page = 1`. → D-5.
**Reproducibility gap:** the cell that produced this CSV is **not currently in the
notebook** — to be re-added so the result regenerates. → D-6.

### E5 — Embedding model  *(n=9; orphan CSV, not refreshed)*
Variable: `embedding_model`. Held: 600/100, top-k=20, reranker on, reranker_top_k=5, max_per_page=1.

| model | Page Recall@5 | MRR | retrieval ms |
|-------|--------------:|----:|-------------:|
| **all-MiniLM-L6-v2** | **1.000** | **0.806** | 19 |
| all-MiniLM-L12-v2 | 0.889 | 0.778 | 33 |

**Finding.** The larger L12 model is both **worse and slower** on this set. No
reason to move off L6-v2.
**Decision:** keep **all-MiniLM-L6-v2**. → D-5. Same reproducibility gap as E4.

### E6 — Generation smoke test (full pipeline)
Config: 600/100, top-k=20, reranker on, reranker_top_k=5, max_per_page=1,
generation `llama3.2:3b` (temp 0). A wiring + failure-mode diagnostic (not an
accuracy score). Re-run over the **43-question set**.

Failure categories (of 43): **Retrieval 4, Generation 2, Citation 2** — the rest
clean. This section's smoke config predates the E7 winner (it uses 600/100 + the
grounded no-cite prompt, not enriched-500/150 + prompt B), so its Citation/Retrieval
counts are *before* the E7/E8 fixes and mainly serve to localize failure types.

**Finding.** Three distinct failure types persist at the baseline config:
retrieval-content misses, over-cautious refusals, and citation leaks. E7 (chunk
overlap + enrichment) addresses the first, E8 (prompt B) the latter two — which is
why the smoke test is kept as a *diagnostic* while E7/E8 define the selected pipeline.

---

### E7 — Content-level metric + feature engineering (the big round)

Page Recall@5 had saturated at 1.0, so it could no longer rank configs. Added a
**content-coverage** metric: fraction of a question's answer terms (ground-truth
tokens ≥5 chars) present in the *delivered* context, plus `cov≥0.5` (how many
answerable questions clear half-coverage). Then swept feature-engineering variants
(all: top_k=20, reranker on, reranker_top_k=5, max/page=1, MiniLM-L6-v2).

Feature-engineering variants built:
- **metadata enrichment** — prepend `Source | Page N | Section: <nearest heading>`
  to each chunk before embedding (headings detected by regex on the page text);
- **structure-aware separators** — split on bullet markers (`\n• `, `\n- `, `\n* `)
  before prose;
- **hybrid BM25+dense** — Reciprocal-Rank Fusion (k=60) of `rank_bm25` and Qdrant.

Results on the **43-question set** (n=39 answerable; `cov` = mean content coverage):

| config | #chunks | Recall@5 | MRR | cov | cov≥0.5 |
|--------|--------:|---------:|----:|----:|--------:|
| plain 600/100 (old baseline) | 50 | 0.974 | 0.853 | 0.732 | 29/39 |
| plain 500/150 | 73 | 0.949 | 0.897 | 0.721 | 29/39 |
| enriched 600/150 | 57 | 0.974 | 0.919 | **0.838** | **34/39** |
| **enriched 500/150 (selected)** | 73 | **1.000** | **0.944** | 0.759 | 31/39 |
| enriched 500/150 + hybrid | 73 | 0.949 | 0.851 | 0.682 | 27/39 |
| enriched 400/150 | 98 | 1.000 | 0.919 | 0.723 | 29/39 |
| enriched 500/200 | 82 | 0.974 | 0.929 | 0.773 | 31/39 |

**Findings.**
1. **Q3 was a chunk-overlap problem, not a feature problem** (found on the original
   set): the ingestion-order list spans a boundary overlap-100 cuts; overlap **150**
   captures it (`Q3cov 0.17 → 1.00`). No enrichment/hybrid helped at overlap 100.
2. **Factors interact — one-at-a-time tuning would mislead.** Metadata enrichment
   *hurt* at overlap 100 but *helped* at overlap 150. Chunk size, overlap and
   enrichment must be tuned jointly.
3. **Hybrid BM25 and structure-aware splitting did not help on this document** —
   hybrid consistently lowered coverage. Kept OFF. Measured, not assumed.
4. **The bigger set exposed a coverage-vs-recall trade-off the n=9 set hid.**
   `enriched 500/150` is retrieval-best (recall **1.000**, MRR **0.944**);
   `enriched 600/150` is delivery-best (coverage **0.838**, cov≥0.5 **34/39**) but
   misses one page (recall 0.974). The selection rule — *best coverage among configs
   with recall@5 = 1.0* — picks **enriched 500/150**; `enriched 600/150` is the
   documented alternative when answer-text coverage matters more than perfect page
   recall.
5. Selected **enriched 500/150** still beats the old baseline across the board:
   MRR 0.853→**0.944**, recall 0.974→**1.000**, coverage 0.732→0.759.

### E8 — Generation prompt tuning (Q5 over-refusal + citation leaks)

Retrieval fixed to the E7 winner. Compared the original prompt (A) with an
answer-preferring prompt (B), over all 43 questions.

| prompt | refusals on answerable | citation leaks | unanswerable refused |
|--------|----------------------:|---------------:|---------------------:|
| A (original) | 9/39 | **31/43** | yes |
| **B (revised)** | **0/39** | **0/43** | yes |

*(n=43 set. On the original n=9 set this was A: 2/9 refusals, 7/10 leaks; B: 0, 0 —
the bigger set confirms the effect and shows A leaks on 31 of 43 questions.)*

**Findings.**
- Prompt A was **self-contradictory**: it told the model to "cite source numbers
  such as [SOURCE 1]" while the pipeline forbids model citations — that is the
  source of the leaks (31 of 43), not model misbehaviour.
- Prompt B (answer if the context supports it even partially; never emit citations;
  refuse only when truly unsupported) removes all refusals and all leaks **without**
  breaking the unanswerable refusal. Keep the code-side citation strip as a backstop.
- **Remaining issue (Q5):** B now *answers* Q5 but with the wrong list
  ("Correctness, relevancy, hallucination, toxicity" — DeepEval-style, not the Ragas
  metrics). The context holds a competing list, so Q5 is a retrieval-**precision**
  problem, not a generation one. Open.

## Selected configuration (current best — updated after E7/E8)

```
chunk_size            = 500          # was 600
chunk_overlap         = 150          # was 100  (fixes Q3 content miss)
chunk_metadata_enrich = True         # NEW: prepend Source|Page|Section header
chunk_separators      = plain ladder # structure-aware gave no gain
hybrid_bm25           = False        # tested; hurt coverage
embedding_model       = sentence-transformers/all-MiniLM-L6-v2
retrieval_top_k       = 20
reranker_enabled      = True
reranker_model        = cross-encoder/ms-marco-MiniLM-L-6-v2
reranker_top_k        = 5
max_chunks_per_page   = 1
generation_prompt     = B (answer-preferring, no-citation, grounded refusal)
generation_model      = llama3.2:3b (Ollama, temp 0) + code-side citation strip
```

Measured on the **43-question set** (39 answerable): Page Recall@5 = 1.000,
MRR = 0.944, content coverage = 0.759, cov≥0.5 = 31/39; generation (prompt B)
0/39 refusals, 0/43 citation leaks, all 4 unanswerable questions refused.

---

## Open issues / next experiments

1. ✅ **Q3 content miss — fixed** by chunk_overlap=150 + enrichment (E7).
2. ◑ **Q5 is now a retrieval-precision problem, not a refusal** (E8). The context
   holds both the Ragas list and a competing DeepEval-style list; the model picks
   the wrong one. → try finer chunks around enumerations, or a precision-oriented
   rerank; measure with answer-term *accuracy*, not just coverage.
3. **Hybrid BM25 / structure-aware splitting** gave no gain on this single doc
   (E7). Re-test if the corpus grows to many documents with exact-match terms.
4. **Close the reproducibility gap (E4, E5).** Re-add the page-diversification and
   embedding-model sweeps as notebook cells.
5. **Ragas evaluation** (faithfulness, answer relevancy, context precision/recall)
   — needs `ragas` installed; the natural successor to the content-coverage proxy.
   *Deferred to production phase — see D-7.*
6. ✅ **Bigger golden set — done** (10 → 43, 2026-10-10). Metrics now move in ~1/39
   steps. Next: push toward the roadmap's full 50, and add answer-term *accuracy*
   (not just coverage) to catch the Q5 wrong-list case.

---

## Generation-step finding (from E6, carried forward)

**The generation step leaks citation markers.** `llama3.2:3b` sometimes echoes the
context's `[SOURCE n | Page m]` labels (e.g. "According to SOURCE 2 | Page 3, …")
despite the prompt forbidding it — the same Citation failure the notebook
classifier flags. The prompt-only approach is **not** sufficient for small local
models; because Python owns citations, the generation step must **strip leaked
markers in code** (a `re.sub` over `[SOURCE n | Page m]`, `[n]`, `Page m`, plus the
dangling "according to …," connective — verified to leave clean answers and the
exact "I don't know" refusal untouched). Recorded here as a requirement for the
generation step; it is a pipeline concern, independent of where generation is later
hosted.

## Production phase — deferred (not current scope)

`app/` is intentionally left empty: wiring the selected pipeline into a package /
FastAPI service is the **production** phase, to be done after the experimental
pipeline is locked. Deferred items (see `decision.md` D-7):

- [ ] `app/` package wired to the final config
- [ ] FastAPI serving layer *(needs `fastapi`/`uvicorn`)*
- [ ] Ragas regression eval in CI *(needs `ragas`)*
