"""The hand-grounded golden set.

Each question carries the ``source`` filename and the 1-based ``source_pages``
it is answerable from (empty for unanswerable questions). Ground-truth strings
use the document's own vocabulary so content-coverage scores them fairly. A
build-time sanity check (``harness.check_grounding``) verifies that each answer's
terms actually appear on its claimed page.

To extend to a new document: add its PDF to ``data/documents/`` and append
questions here with ``source`` set to the new filename.
"""
from __future__ import annotations

UNKNOWN_ANSWER = "I don't know based on the provided documents."

# Known source documents (filenames as they appear in data/documents/).
ROADMAP = "Shashwat_AI_ML_Project_Roadmap.pdf"

golden_set: list[dict] = [
    # ---- Page 1: career direction + Project 1 DocuMind ----
    {"source": ROADMAP, "question": "What career direction does the roadmap recommend?", "ground_truth": "Applied AI/ML Engineering rather than a deep-research ML Scientist track.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "What technologies are listed for Project 1, DocuMind?", "ground_truth": "FastAPI, LangChain or LlamaIndex, Qdrant or ChromaDB, sentence-transformers, Redis, Ragas, Langfuse, Docker, and GitHub Actions.", "source_pages": [1], "type": "multi_context"},
    {"source": ROADMAP, "question": "What is the ingestion pipeline order for DocuMind?", "ground_truth": "PDF/text loader, chunking, embeddings, then vector store.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "What retrieval stages are proposed before the LLM in DocuMind?", "ground_truth": "Similarity search followed by a lightweight Cross-Encoder reranker, then the context is passed to the LLM.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "Which Ragas metrics are named for the DocuMind golden set?", "ground_truth": "Faithfulness, answer relevancy, context precision, and context recall.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "What two changes does the roadmap say will move the needle fastest?", "ground_truth": "Standardize on Python for all AI portfolio work, and move from trained a model to shipped a system.", "source_pages": [1], "type": "multi_context"},
    {"source": ROADMAP, "question": "Which FastAPI endpoints are proposed for DocuMind?", "ground_truth": "The /ingest, /query, and /feedback endpoints.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "What chunking approaches does DocuMind suggest experimenting with?", "ground_truth": "Fixed-size chunking versus semantic chunking.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "What is each DocuMind pipeline stage wrapped in for tracing?", "ground_truth": "Langfuse trace spans, so latency and token cost per stage are visible.", "source_pages": [1], "type": "factual"},
    {"source": ROADMAP, "question": "Which existing ML projects prove the candidate's modeling ability?", "ground_truth": "The Hotel Booking and CDSS projects.", "source_pages": [1], "type": "factual"},

    # ---- Page 2: DocuMind golden set + stretch goals + Project 2 ----
    {"source": ROADMAP, "question": "How large should the DocuMind golden set be?", "ground_truth": "About 30 to 50 questions.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What stretch-goal technique is suggested for vague user questions?", "ground_truth": "Query rewriting or HyDE, Hypothetical Document Embeddings.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What does multi-tenant document isolation prevent?", "ground_truth": "User A's documents never leak into User B's answers.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "Why does the roadmap suggest hybrid search?", "ground_truth": "Pure vector search can miss exact-match terms such as product codes or names.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What tool is suggested to load-test the DocuMind API?", "ground_truth": "Locust, to report concurrent users at a p95 latency.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What nodes are in the Project 2 LangGraph agent?", "ground_truth": "Planner, Researcher, Writer, and Critic.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What observability technologies are proposed for Project 2?", "ground_truth": "OpenTelemetry and Langfuse, with cost and latency shown per node.", "source_pages": [2], "type": "multi_context"},
    {"source": ROADMAP, "question": "What protocol does Project 2 use for tool calling instead of hardcoded LangChain tools?", "ground_truth": "MCP, the Model Context Protocol.", "source_pages": [2], "type": "factual"},
    {"source": ROADMAP, "question": "What does the Critic node do in the AgentOps graph?", "ground_truth": "Checks the draft against the sources and sends it back to the Writer if unsupported claims are found.", "source_pages": [2], "type": "factual"},

    # ---- Page 3: Project 2 stretch goals ----
    {"source": ROADMAP, "question": "What long-term memory stretch goal is proposed for AgentOps?", "ground_truth": "Store past research sessions in a vector store so the agent can recall earlier work.", "source_pages": [3], "type": "factual"},
    {"source": ROADMAP, "question": "What is model routing in Project 2?", "ground_truth": "Use a small cheap model for the Planner and a stronger model only for the Writer, then report cost savings.", "source_pages": [3], "type": "factual"},
    {"source": ROADMAP, "question": "What guardrail addresses runaway agent loops?", "ground_truth": "A hard token or cost budget per session with a kill-switch node.", "source_pages": [3], "type": "factual"},

    # ---- Page 4: Project 3 serving/monitoring + Project 4 PromptBench ----
    {"source": ROADMAP, "question": "Which tools provide metrics and dashboards in Project 3?", "ground_truth": "Prometheus and Grafana.", "source_pages": [4], "type": "factual"},
    {"source": ROADMAP, "question": "What tool does Project 3 use for drift detection?", "ground_truth": "Evidently AI, for data and prediction drift.", "source_pages": [4], "type": "factual"},
    {"source": ROADMAP, "question": "What is Project 4 called and what does it do?", "ground_truth": "PromptBench, an LLM evaluation and regression testing framework, unit tests for prompts.", "source_pages": [4], "type": "multi_context"},
    {"source": ROADMAP, "question": "Which metrics does PromptBench score?", "ground_truth": "Correctness, relevancy, hallucination, and toxicity.", "source_pages": [4], "type": "factual"},

    # ---- Page 5: Project 4 architecture ----
    {"source": ROADMAP, "question": "What format defines a PromptBench test case?", "ground_truth": "YAML or JSON: a prompt template plus input variables and an expected reference answer.", "source_pages": [5], "type": "factual"},
    {"source": ROADMAP, "question": "What stretch goal validates the LLM-as-judge itself?", "ground_truth": "Have a human score a sample and report inter-rater reliability agreement with the judge model.", "source_pages": [5], "type": "factual"},

    # ---- Page 6: Project 5 SupportGraph ----
    {"source": ROADMAP, "question": "What does SupportGraph do when the model is unsure?", "ground_truth": "Escalates the ticket to a human-review queue instead of auto-sending.", "source_pages": [6], "type": "factual"},
    {"source": ROADMAP, "question": "What privacy step runs on incoming tickets in SupportGraph?", "ground_truth": "A PII-redaction pass using regex or Presidio.", "source_pages": [6], "type": "factual"},

    # ---- Page 7: Project 6 MCP server + fit-together ----
    {"source": ROADMAP, "question": "Give examples of MCP tools Project 6 might expose.", "ground_truth": "search_notes, get_repo_issues, and predict_booking.", "source_pages": [7], "type": "factual"},
    {"source": ROADMAP, "question": "Which three projects does the roadmap recommend building first?", "ground_truth": "Projects 1, 2, and 4.", "source_pages": [7], "type": "factual"},

    # ---- Page 8: trending concepts ----
    {"source": ROADMAP, "question": "Who introduced the Model Context Protocol?", "ground_truth": "Anthropic.", "source_pages": [8], "type": "factual"},
    {"source": ROADMAP, "question": "What is semantic caching?", "ground_truth": "Caching LLM responses by the meaning of a query via embedding similarity, not exact string match.", "source_pages": [8], "type": "factual"},
    {"source": ROADMAP, "question": "What lightweight fine-tuning techniques are mentioned?", "ground_truth": "LoRA and QLoRA, parameter-efficient fine-tuning.", "source_pages": [8], "type": "factual"},

    # ---- Page 9: DSA + system design ----
    {"source": ROADMAP, "question": "What vector index choices are named for RAG at scale?", "ground_truth": "The vector index choice between HNSW and IVF.", "source_pages": [9], "type": "factual"},
    {"source": ROADMAP, "question": "Which practice problem lists does the DSA plan recommend?", "ground_truth": "The NeetCode 150 or the Blind 75.", "source_pages": [9], "type": "factual"},

    # ---- Page 10: sources + 12-week roadmap ----
    {"source": ROADMAP, "question": "What book is recommended for ML system design?", "ground_truth": "Designing Machine Learning Systems by Chip Huyen.", "source_pages": [10], "type": "factual"},
    {"source": ROADMAP, "question": "What is planned for weeks 1 and 2 of the roadmap?", "ground_truth": "Project 1, DocuMind RAG API, plus DSA topics: Arrays, Hashing, Two Pointers, and Binary Search.", "source_pages": [10], "type": "multi_context"},

    # ---- Unanswerable (grounding guard); source None = not in any document ----
    {"source": None, "question": "What is the population of Mars according to this document?", "ground_truth": UNKNOWN_ANSWER, "source_pages": [], "type": "unanswerable"},
    {"source": None, "question": "What monthly salary does this roadmap guarantee?", "ground_truth": UNKNOWN_ANSWER, "source_pages": [], "type": "unanswerable"},
    {"source": None, "question": "Which cloud provider (AWS, GCP, or Azure) does the roadmap require?", "ground_truth": UNKNOWN_ANSWER, "source_pages": [], "type": "unanswerable"},
    {"source": None, "question": "How many GitHub stars does the DocuMind repository have?", "ground_truth": UNKNOWN_ANSWER, "source_pages": [], "type": "unanswerable"},
]
