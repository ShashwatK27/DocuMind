# DocuMind — Decision Log

Decisions taken while fixing the notebook and driving the project toward
completion: **what** was decided, **why**, and the **alternatives** rejected.
Companion to [`experiments.md`](experiments.md) (the measured results these
decisions rest on).

Format: each entry is a decision, its rationale, and what was considered instead.

---

### D-1 — Ollama runs behind a connection *probe*, not a bare flag
**Decision.** Keep generation enabled by default (`RUN_LLM_EVALUATION = True`) but
construct the LLM only after a one-line probe (`candidate_llm.invoke("ping")`);
on any failure, print a notice and fall back to `llm = None`
(retrieval-only). **Why.** The notebook previously claimed "generation disabled by
default" in prose while the flag was `True` — a "Run All" with Ollama down would
raise deep inside the evaluation loops. A probe makes the degraded path graceful
and keeps the retrieval experiments runnable on any machine.
**Rejected:** flipping the flag to `False` (hides the capability and contradicts
the roadmap, which wants generation measured); catching errors per-call (noisier,
same effect).

### D-2 — Patch the notebook via a verified JSON script, not `NotebookEdit`
**Decision.** Edit `rag_1.ipynb` with a Python script that matches an expected
marker string in each target cell before replacing it, and back up to
`rag_1.ipynb.bak` first. **Why.** The notebook (~41k tokens with outputs) exceeds
the `Read` tool limit, and `NotebookEdit` requires a prior in-conversation `Read`.
The marker-checked script refuses to edit a drifted cell, which is actually safer
than a blind positional edit. Validated afterwards with `nbformat.validate`.
**Rejected:** clearing outputs to shrink the file enough to `Read` (loses the
committed outputs needlessly).

### D-3 — Chunk selection is data-driven: 600/100, tie broken toward smaller
**Decision.** Compute `SELECTED_CHUNK_SIZE/OVERLAP` from the experiment table by
**max MRR, then max recall, then smallest chunk**, instead of hard-coding 600/100.
**Why.** 600/100 and 800/150 tie exactly (MRR 0.778, recall 0.889). The tie-break
toward the smaller chunk is principled: smaller units are more precise, carry less
irrelevant text into the reranker, and cost less context. The rule happens to
reproduce the original 600/100 — but now the choice is *derived and reproducible*,
not asserted. **Why it matters:** every downstream sweep (top-k, reranker, smoke
test) now inherits the selection by variable, so re-running with a different doc
can't silently leave a stale 600 behind.
**Rejected:** picking by recall alone (all tied at top-k=10, undecidable); picking
the largest chunk for fewer vectors (worse MRR).

### D-4 — Ship top-k=20 with the reranker (revised on the n=39 set)
**Decision.** Standardize on `retrieval_top_k = 20` and `reranker_enabled = True`.
**Why.** E3 on the 43-question set shows the reranker is **broadly** valuable — it
lifts MRR at every top-k (0.729→0.893 at k=10, 0.729→0.915 at k=20) and recall at
k=20 (0.923→0.974), and it cannot recover a page the candidate pool lacks, so it
pairs with a wide pool. Top-k=20 is what the validated E7 winner (enriched-500/150)
uses to reach recall 1.0. **Caveat found by the bigger set:** on the *plain* 600/100
setup, top-k=15 slightly beats top-k=20 (recall 1.0 vs 0.974) at lower latency — the
n=9 claim that recall improved "only at top-k=20" was a small-sample artifact.
Top-k=15 is logged as a cheaper candidate to A/B. **Rejected:** reranker off
(clearly lower MRR at every k now); committing to top-k=15 before A/B-ing it with
enrichment held in.

### D-5 — `max_chunks_per_page = 1`; keep MiniLM-L6-v2
**Decision.** Cap final context at one chunk per page; keep the L6-v2 embedder.
**Why.** E4: capping at 1 lifts MRR 0.806→0.815 at no recall cost. E5: the larger
L12 embedder is both worse (MRR 0.778, recall 0.889) and slower (33 vs 19 ms). Both
are small-sample effects but point the same way and cost nothing.
**Rejected:** `max_per_page=2` (the old default; slightly lower MRR); L12-v2
(no upside found).

### D-6 — Treat the E4/E5 CSVs as a reproducibility gap to close
**Decision.** Record that `page_diversification_experiment_results.csv` and
`embedding_model_experiment_results.csv` exist from a prior run but their
generating **cells are not in the current notebook**, and plan to re-add them.
**Why.** Results that can't be regenerated from the committed notebook aren't
trustworthy long-term. Flagging beats silently relying on orphaned CSVs.
**Rejected:** deleting the CSVs (they're real signal); silently trusting them.

### D-7 — Keep `app/` empty; current scope is the experimental pipeline
**Decision.** `app/` stays an empty scaffold. The current phase is developing the
**pipeline in the notebook** — best hyperparameters and feature engineering — not
productionizing it. Wiring the selected config into a package / FastAPI service and
adding Ragas CI is the **production phase**, done only after the pipeline is locked.
**Why.** This is the user's explicit direction: `app/` was deliberately left empty
because it is production work. Experiment results must settle before freezing them
into product code, or the product encodes a moving target.
**History.** An earlier pass in this session filled `app/` from the then-current
config; it was reverted to empty on this clarification. The pipeline logic and the
selected config remain captured in the notebook, `experiments.md`, and D-3…D-5.
**Rejected:** building the package now (premature — pipeline not yet optimized).

### D-8 — Use `venv/`, not `.venv/`
**Decision.** Run everything with `venv/Scripts/python.exe`. **Why.** The repo has
two environments; `.venv/` has only pip+setuptools while `venv/` holds the full
stack (torch, sentence-transformers, qdrant, langchain-*) and is the notebook's
declared kernel ("venv (3.11.9)"). `nbclient` was added to `venv/` to execute the
notebook headlessly. **Rejected:** `.venv/` (empty); creating a third env.

### D-9 — The generation step must strip leaked citations in code, not by prompt
**Decision (pipeline requirement, not app code).** The generation step must
post-process model output to remove any `[SOURCE n | Page m]`, `[n]` or `Page m`
labels it emits, and tidy the dangling "according to …," connective. **Why.**
Running generation showed `llama3.2:3b` echoing the context's source labels despite
an explicit "do not cite" instruction. Because the project invariant is *Python
owns citations*, relying on the prompt is both unreliable (small models disobey)
and architecturally wrong. A deterministic `re.sub` pass, verified to leave clean
answers and the exact "I don't know" refusal untouched, is the fix. This is
recorded as a requirement of the generation step; it applies wherever generation
ends up hosted. **Rejected:** stronger prompt wording only (observed to still
leak); few-shot examples (adds latency/tokens for a formatting concern code can
enforce directly).

### D-10 — Add a content-coverage metric; page-recall alone can't tune the pipeline
**Decision.** Introduce **content coverage** = fraction of a question's ground-truth
answer terms (tokens ≥5 chars) present in the *delivered* context, reported with
`cov≥0.5` over the answerable set, alongside page-recall and MRR. **Why.** At the
prior config page-recall was pinned at 1.0 and MRR near-flat, so neither could
distinguish a better pipeline from a worse one — yet answers were still wrong
(Q3). Coverage is a cheap, deterministic proxy for "did we actually deliver the
evidence," and it immediately separated configs (0.63–0.88) and localized the Q3
miss. **Rejected:** going straight to Ragas (needs install + LLM-judge latency;
the proxy is enough to drive hyperparameter choice and is a stepping stone to it);
trusting page-recall (saturated, misleading).

### D-11 — New selected config: enriched 500/150; hybrid & structure rejected on data
**Decision.** Move the retrieval config to **chunk 500 / overlap 150 with metadata
enrichment** (prepend `Source | Page | Section`), keep dense-only retrieval (no
BM25 hybrid), plain separators, top_k=20, reranker on, reranker_top_k=5,
max/page=1. **Why.** E7 measured this as the best on every meaningful axis —
coverage 0.727→0.884, cov≥0.5 7/9→9/9, MRR 0.815→0.870, recall held at 1.0, Q3
content 0.17→1.0. Two sub-decisions matter: (a) overlap 150 (not 100) because the
Q3 list spans a chunk boundary; (b) enrichment is kept **only because** it was
measured to help *at overlap 150* — it hurt at overlap 100, so the factors were
tuned jointly, not independently. **Rejected with data:** BM25 hybrid (lowered
coverage 0.884→0.783 on this single doc), structure-aware separators (neutral),
larger/smaller chunks and other overlaps (all dominated). These stay available for
a larger, multi-document corpus.

### D-12 — Adopt the answer-preferring generation prompt; fix the citation contradiction
**Decision.** Replace the strict prompt with **Prompt B**: answer if the context
supports it even partially, never emit citations/source labels, refuse (exact
UNKNOWN string) only when the context is genuinely irrelevant. Keep the code-side
citation strip (D-9) as a backstop. **Why.** E8 showed the old prompt both
over-refused and leaked citations heavily — the latter because it literally
asked the model to "cite source numbers such as [SOURCE 1]" while the pipeline
forbids model citations (on the 43-question set A leaks on 31/43 and refuses 9/39
answerable). Prompt B gives 0/39 refusals and 0/43 leaks while still refusing every
unanswerable question. **Rejected:** keeping Prompt A (self-
contradictory); removing the refusal instruction entirely (breaks the unanswerable
guard). **Known residual:** Prompt B answers Q5 with the wrong list — a retrieval
precision issue tracked in `experiments.md`, not a prompt fix.

### D-13 — Expand the golden set to 43 questions, grounded and sanity-checked
**Decision.** Grow the golden set from 10 → **43 questions (39 answerable, 4
unanswerable)**, authored across all 10 PDF pages, and add a build-time sanity
check: every answerable question's ground-truth terms must actually appear on its
claimed source page. **Why.** At n=9 every metric moved in 1/9 steps, so single-
question flips masqueraded as signal and the "winner" rested on thin evidence — the
roadmap itself asks for 30–50 questions. The sanity check caught **2 real authoring
bugs** (HNSW/IVF acronyms too short to match the term filter; "leaking" vs the
document's "leak"), which were reworded to the document's own vocabulary.
**Outcome.** At n=39 the selected config `enriched-500/150` **held** (recall 1.0,
MRR 0.944) and the data revealed a genuine **coverage-vs-recall trade-off** with
`enriched-600/150` (coverage 0.838 vs 0.759) that the small set had hidden — so the
bigger set changed our *understanding* even though it confirmed the *choice*.
**Selection rule kept:** best content coverage among configs with Page Recall@5 =
1.0 → `enriched-500/150`; `enriched-600/150` documented as the coverage-first
alternative. **Rejected:** keeping n=9 (misleadingly coarse); auto-generating
questions with an LLM (unverified ground truth — hand-grounding is the point of a
golden set). **Next lever:** the roadmap's full 50, and answer-term *accuracy* (not
just coverage) to catch the Q5 wrong-list case.

### D-14 — Production phase: build `app/` from the locked config
**Decision.** With the pipeline settled, fill the `app/` scaffold with a FastAPI
service that **encodes the selected configuration** and exposes the roadmap's
endpoints (`/ingest`, `/query`, `/feedback`, plus `/health` and a minimal demo
UI). Ship a `Dockerfile` + `docker-compose.yml` (API + Qdrant server; Ollama on
the host), a README that leads with the measured results, and `.env.example`.
**Why.** This is the explicit next step the user asked for ("convert it to a
system"), and it is the single highest-value move for the project as a portfolio
piece — the roadmap's whole thesis is *"ship a system, not a notebook."* D-7
deliberately kept `app/` empty *until the pipeline was locked*; it now is (D-11,
D-12, confirmed on n=43 in D-13), so freezing it into product code no longer
encodes a moving target. **Fidelity to the experiments:** every hyperparameter in
`app/config.py` carries a comment citing the decision that set it, so the service
and the notebook cannot silently diverge. The notebook invariants are honored in
code — Python owns citations (D-9), Prompt B with the strip backstop (D-12),
the Ollama connection probe with graceful retrieval-only fallback (D-1).
**Vector store:** the app prefers a Qdrant *server* (via `DOCUMIND_QDRANT_URL`,
set by compose) over a local on-disk store, which sidesteps the OneDrive sqlite
lock seen in dev. **Rejected:** a thin notebook-export script (loses the API and
the deployability that make it a "system"); bundling Ollama into compose (heavy
model pull; host Ollama keeps it on the user's GPU and matches the local-first
assumption); re-implementing the pipeline rather than lifting the validated
`fe_lib`/notebook logic (would risk drift from the measured results).

### D-15 — Evaluate on a multi-document corpus; hybrid search now has evidence
**Decision.** Grow the evaluation universe from the single roadmap PDF to a
**four-document corpus** — the roadmap plus three public arXiv papers (Transformer,
BERT, RAG), fetched reproducibly by `scripts/fetch_corpus.py` and *not* committed
(third-party content; gitignored, roadmap kept) — and hand-ground **20 new
questions** across the papers (golden set 43 → 63, 59 answerable). Run everything
through the committed `eval/` harness, which reuses the real `app` pipeline.
**Why.** A single document made page-recall saturate at 1.000, which the n=43
work already flagged as hiding signal; a realistic system is multi-document, and
the roadmap's own thesis is to ship one. **Outcome.** Recall@5 fell 1.000 → 0.932
and MRR 0.944 → 0.852 — not a regression but the metric finally having room to
move. The **4 genuine misses are all exact-match cases** (*Ragas*, *Adam*,
*FAISS*, *HNSW*): a rare, decisive token sits on the correct page, but dense
cosine similarity ranks a semantically-similar passage (often from the RAG paper)
above it. **This overturns the single-doc reading of D-11.** Hybrid BM25+dense
was rejected there because, on one document with no lexical competitors, it only
lowered coverage. On a multi-document corpus the exact-match failures it is
designed to fix actually occur — so hybrid search (and/or metadata/source
filtering) is now the **next experiment**, to be measured on this corpus before
adopting. **Also decided:** credit a Recall@k hit on *any* page that genuinely
states the answer — three new questions were re-grounded to two pages each after
inspection (e.g. BERT `[CLS]`/`[SEP]` on pages 3–4), which is correct recall
practice, not metric-gaming; the four exact-match misses were left standing as
real signal. **Correctness fix shipped alongside:** page diversification now keys
on `(source, page)` so two documents' page 1 are not merged (committed with the
retriever refactor). **Rejected:** LLM-generated questions for the new docs
(unverified ground truth — the questions were hand-grounded and checked against
the page text); committing the PDFs (redistribution of third-party papers — a
fetch script is cleaner and keeps the repo small).

### D-16 — Adopt hybrid retrieval (BM25-augmented dense); naive RRF rejected
**Decision.** Turn hybrid retrieval **on by default** in the service, implemented
as **candidate augmentation**: take the dense top-k, add BM25's top-k lexical
matches that aren't already present, and let the existing cross-encoder rerank
the union. **Why.** Measured on the 4-document corpus (D-15's open experiment),
this beats dense on every axis with no regressions — Recall@5 0.932 → **0.949**,
MRR 0.852 → **0.890**, coverage 0.712 → **0.744** — and recovers exact-match
misses (e.g. the roadmap's "vector index … for RAG at scale", now rank 1) without
displacing any previously-correct answer. **This reverses D-11 for the
multi-document regime** (D-11 rejected hybrid on a single doc, where there were no
lexical competitors; that rejection still stands for single-doc corpora).
**Rejected — naive Reciprocal-Rank Fusion:** the textbook RRF variant was tried
first and was a *wash* (Recall@5 +0.000, coverage −0.030, cov≥0.5 −0.085). Because
RRF re-truncates the fused list to k, BM25's lexical hits evicted correct dense
chunks from the candidate pool *before the reranker could see them* — it recovered
2 misses but introduced 2 new ones. Candidate augmentation keeps every dense
chunk and only *adds* options, so a strong reranker can only benefit. Both results
are kept in `experiment/hybrid_vs_dense_results.csv`. **Serving note.** BM25 needs
the chunk texts, which Qdrant doesn't return at query time, so the service keeps a
process-wide BM25 index rebuilt by scrolling the collection (survives restarts)
and invalidated on ingest; the eval harness passes the corpus in directly.
**Still missed (3):** the RAG-paper FAISS/HNSW page, the Transformer optimizer
page, and the roadmap "Ragas" question — candidates for the next lever
(query-aware weighting, or an answer-term accuracy metric).

### D-17 — Query-aware fusion at the final ranking (α = 0.15)
**Decision.** After the cross-encoder scores the candidate pool, rank not on the
raw reranker score but on a **query-normalized blend** `reranker_z + α · bm25_z`
with **α = 0.15** (config `fusion_alpha`; 0.0 disables, only active on the hybrid
path). **Why.** D-16's augmentation put the exact-match chunks *into* the pool,
but a per-query diagnostic showed the reranker still placed two of them at ~rank 7
— just outside the diversified top-5 — even though BM25 ranked them 3rd and 8th.
The signal was present; only the final ordering was wrong. Standardizing both
signals per query and adding a light lexical term lets a chunk *both* models like
rise, without either dominating. **Outcome (4-doc corpus).** Recall@5 0.949 →
**0.983**, MRR 0.890 → **0.897**, coverage 0.744 → **0.754**, misses 3 → **1**,
with **no regressions** — recovering the Transformer *Adam optimizer* page and the
RAG *FAISS/HNSW* page. α was swept {0.1…1.0}: 0.15 is the knee, and α ≥ 0.5 erodes
MRR while α = 1.0 displaces two other exact-match answers (recall back to 0.949),
confirming this is a light-touch nudge, not a lexical takeover. **Rejected —
raising `reranker_top_k` past 5:** it would lift Recall@5 by handing the LLM more
chunks, but that games the metric (we *measure* at 5) and dilutes precision rather
than fixing the ordering. **Rejected — relabelling the last miss:** the roadmap
"Which Ragas metrics … for the golden set?" question still misses because the
query points at page 2 (the golden-set paragraph, which genuinely names Ragas
metrics) while the full four-metric enumeration is on page 1. Page 2 is a
defensible answer, so moving the ground-truth label to make it pass would be
gaming; left standing. **Distinct from the D-16 rejection of RRF:** that was
rank-fusion used for *candidate selection* (it evicted dense chunks before
rerank); this is score-fusion at the *final* stage over a pool that already keeps
every dense chunk, so nothing is evicted.

### D-18 — Unit tests + CI with an eval regression gate
**Decision.** Add a fast, network-free `pytest` suite plus a GitHub Actions
workflow (`.github/workflows/ci.yml`) that runs on every push/PR with two jobs:
**unit tests** (pipeline invariants) and an **eval regression gate**
(`scripts/check_eval.py`) that fetches the corpus, runs the real `eval` harness,
and fails the build if Page Recall@5, MRR or coverage drops below a floor.
**Why.** The repo is now public and the roadmap's own thesis (p2) is that
automated, on-every-push quality checks are "what turns *I built RAG* into *I
built a RAG system with regression-tested quality*." A green badge on the repo is
the visible proof. **What the tests cover.** Pure, deterministic logic with no
model downloads: metadata enrichment prefix + split (chunker), `_tok`/BM25
ordering, `_zscore`, the D-17 fusion maths (exact formula + α-scaling promotion),
`_diversify` (1/page cap, top-k truncation, source-aware), the citation strip
(leak removal + exact-refusal preservation) and the no-chunks refusal path, and
the config defaults (so a stray edit can't silently change the served pipeline).
**Thresholds.** Recall@5 ≥ 0.95, MRR ≥ 0.88, coverage ≥ 0.72 — a margin below the
measured 0.983 / 0.897 / 0.754 (D-17) to tolerate library drift without masking a
real regression. **Design choices.** Unit tests avoid the embedder/cross-encoder
so they run in ~12 s and need no network; the heavier eval job caches the HF
weights and is a separate job so unit tests give fast feedback. The eval gate
needs no Ollama (it scores retrieval only). `requirements-dev.txt` layers `pytest`
on the pinned service deps. **Rejected — asserting exact metric values:** brittle
across library versions; a floor catches regressions without false alarms.
**Rejected — a store-backed retrieval integration test in the unit suite:** it
would pull model weights and the arXiv PDFs, making the "fast" gate slow and
network-bound; that coverage lives in the eval job instead.

---

## Standing assumptions (challenge any of these if wrong)

- The evaluation universe is the 63-question golden set over a four-document corpus
  (roadmap + three arXiv papers); 59 answerable, so metrics now move in ~1/59 steps
  (D-15). The notebook still runs the older single-doc harness; `eval/` is the
  source of truth going forward.
- Local-first: generation stays on Ollama `llama3.2:3b`; no hosted API calls.
- Python owns all citation/metadata; the LLM only drafts prose (notebook invariant,
  carried into `app/`).
