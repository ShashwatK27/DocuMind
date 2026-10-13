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

---

## Standing assumptions (challenge any of these if wrong)

- The 10-question golden set and the single roadmap PDF are the evaluation universe
  for now; metrics are directional given n=9 answerable.
- Local-first: generation stays on Ollama `llama3.2:3b`; no hosted API calls.
- Python owns all citation/metadata; the LLM only drafts prose (notebook invariant,
  carried into `app/`).
