The repository is TaxPilot, a Python multi-agent tax-preparation copilot for Indian income tax returns. Its core design is: LLMs read and explain documents; a deterministic Python engine computes the actual rupee values and attaches the rule behind each figure; the LangGraph pipeline pauses for human review when audited thresholds or unresolved facts need attention. The entrypoint is the FastAPI app in `src/taxpilot/api/main.py`, which serves the static web UI in `web/` and exposes the JSON API.

The real execution path is:

1. HTTP request enters FastAPI.
2. `pipeline.prepare()` builds a LangGraph workflow and runs it.
3. Document ingestion loads files from disk or inline payloads.
4. Each document is classified, extracted, normalized, and fed into a `TaxState`.
5. The graph does:
   - intake guardrail
   - extraction
   - classification
   - deduction research via BM25 knowledge retrieval
   - deterministic tax computation
   - audit scoring
   - optional human review interrupt
   - report generation
   - guardrails
   - finalization
6. The API returns a `ReturnResponse` with the final tax return, report, citations, audit, guardrail results, and review metadata.

Start-to-finish file and code map

- `pyproject.toml`
  - Declares the project as `taxpilot`
  - Scripts:
    - `tax-ingest = "taxpilot.knowledge.ingest:main"`
    - `tax-api = "taxpilot.api.main:run"`
  - Dependencies:
    - `langgraph`
    - `fastapi`
    - `uvicorn`
    - `anthropic`
    - `pydantic`, `pydantic-settings`, `python-dotenv`
    - optional OCR extras for PDFs/images

- `src/taxpilot/config.py`
  - `Settings` reads `.env`
  - Controls:
    - `llm_provider` = `"anthropic"` or `"gemini"`
    - `tax_year` = 2024
    - `corpus_dir`, `documents_dir`
    - review thresholds
    - feature flags: `enable_rag`, `enable_audit`, `enable_review`, `enable_guardrails`
  - `get_settings()` is a cached singleton configuration access point.

- `src/taxpilot/models.py`
  - This is the schema layer shared across the system.
  - Key classes:
    - `SourceDocument`
    - `ExtractionResult`
    - `ExtractedField`
    - `TaxpayerProfile`
    - `IncomeItem`
    - `DeductionCandidate`
    - `Citation`
    - `TaxReturn`
    - `AuditRisk`
    - `ReviewItem`
    - `Correction`
    - `GuardrailOutcome`
    - `TraceStep`
    - `ReturnResponse`
  - Important invariant: money is carried as integers in whole rupees; no LLM output is allowed to become an actual tax figure.

- `src/taxpilot/api/main.py`
  - `PrepareRequest` and `ResumeRequest` define the API schema.
  - `lifespan()` runs when the app starts:
    - `get_settings()`
    - `setup_logging()`
    - `enable_system_trust_store()`
    - `pipeline.get_graph()`
  - App mounts the web UI at `/static` and serves `index.html` at `/`
  - Routes:
    - `GET /health`
    - `POST /prepare`
    - `GET /pending/{thread_id}`
    - `POST /resume`
  - `_inline_documents()` converts incoming JSON docs into `SourceDocument` entries.

- `src/taxpilot/graph/pipeline.py`
  - This is the public orchestration layer.
  - `prepare(documents=None, documents_dir=None, thread_id=None, regime=None, use_llm=True)`
    - If no docs passed, it resolves `documents_dir` from settings and calls `load_documents(directory)`.
    - `graph = graph or get_graph()`
    - `thread_id = thread_id or uuid.uuid4().hex`
    - Runs:
      - `graph.invoke(initial_state(documents, regime or "", use_llm), config=_config(thread_id))`
    - Converts graph output to `ReturnResponse` via `_to_response(...)`
  - `resume(thread_id, corrections=...)`
    - Reads existing graph state using `graph.get_state(config)`
    - Calls `graph.invoke(Command(resume=payload), config=config)`
  - `pending_review(thread_id)`
    - Inspects LangGraph interrupts and returns the review payload.
  - `_to_response()` packages:
    - `tax_return`
    - `audit`
    - `report`
    - `guardrails`
    - `awaiting_review`
    - `review_items`
    - `trace`
    - token counts and latency

- `src/taxpilot/graph/builder.py`
  - This compiles the actual workflow graph.
  - `build_graph()` adds nodes:
    - guardrail_intake
    - extract
    - classify
    - research
    - calculate
    - summarize
    - audit
    - review_gate
    - report
    - guardrail_output
    - finalize
  - `after_intake(state)`:
    - if `state["blocked"]`, route to `finalize`
    - otherwise route to `extract`
  - `after_review(state)`:
    - if `state["recompute"]`, route back to `calculate`
    - otherwise route to `report`
  - `get_graph()` is the process-wide cached compiled graph.
  - The graph uses `MemorySaver` with a custom `JsonPlusSerializer` so pydantic models survive a review resume.

- `src/taxpilot/graph/state.py`
  - Defines `TaxState` as a `TypedDict`.
  - Fields are populated across the run:
    - `documents`
    - `extractions`
    - `profile`
    - `income`
    - `regime_override`
    - `use_llm`
    - `deductions`
    - `tax_return`
    - `audit`
    - `review_required`
    - `classify_notes`
    - `review_items`
    - `reviewed`
    - `corrections`
    - `recompute`
    - `report`
    - `llm_summary`
    - `guardrails`
    - `blocked`
    - `block_message`
    - `trace`
    - `input_tokens`
    - `output_tokens`
  - `initial_state(...)` seeds all empty lists and defaults.

Exact execution sequence

1. App bootstrap
   - `tax-api` entry runs `taxpilot.api.main:run()`
   - `run()` starts uvicorn on the configured host/port.
   - At startup, `lifespan()` compiles the graph as a one-time cost.

2. Prepare request
   - Client POSTs to `/prepare`
   - Request may contain:
     - `documents` inline
     - `documents_dir`
     - `thread_id`
     - `regime` override
     - `use_llm`
   - If `documents` is missing, the app loads files from `settings.documents_dir`.

3. Document ingestion
   - `src/taxpilot/intake/loader.py`
   - `load_documents(directory)`
     - iterates directory entries
     - calls `load_document(path)`
     - `load_document()` calls `extract_text(path)` from `intake/ocr.py`
   - `extract_text(path)`
     - `.txt/.md/.json/.csv` -> read as plain text
     - `.pdf` -> uses `pypdf` if installed
     - images -> OCR via `pytesseract` + `PIL`
     - raises `OcrUnavailableError` if OCR support is missing for scanned files
   - `classify_doc_type(text, filename)`:
     - checks a priority marker list like:
       - “form 16”
       - “form 26as”
       - “80c”
       - “interest certificate”
       - “rent receipt”
       - “home loan”
     - returns a `DocType` enum
   - Constructed object:
     - `SourceDocument(id, filename, doc_type, text, ocr_confidence)`
   - This yields a list of document objects as the first concrete data payload.

4. Initial graph state
   - `initial_state(documents, regime_override, use_llm)` populates `TaxState`
   - The graph is invoked with that state and a thread checkpoint config.

5. Guardrail intake
   - `guardrail_intake_node` in `src/taxpilot/graph/nodes.py`
   - This calls the guardrail pipeline from `src/taxpilot/guardrails/pipeline.py`
   - `Guardrails.check_documents(...)`
     - rejects empty submissions
     - rejects oversized document sets
     - blocks text injection patterns
     - inventories PII
     - optionally calls LLM intake review to verify the submission is actually tax documents
   - If blocked:
     - `after_intake` routes to `finalize`
     - the pipeline ends early.

6. Extraction
   - `extract_node` in `graph/nodes.py`
   - It uses the extractor in `src/taxpilot/extraction/extractor.py`
   - `Extractor.extract(doc)`
     - if LLM available: `_extract_llm(doc)`
     - else: `_extract_heuristic(doc)`
   - `_extract_llm()`
     - builds a prompt with document type + text
     - calls `self.llm.complete_json(...)`
     - expects a JSON schema with `fields`
     - canonicalizes each label with `canonicalize()`
     - coerces numeric strings with `_coerce()`
     - creates `ExtractedField(name, value, confidence, source_doc, box)`
   - `_extract_heuristic()`
     - scans each line with regex:
       - `_LINE` matches `Label: value`
       - `_NUMBER` detects money-like strings
     - turns values into numeric floats where applicable, otherwise leaves them as text
   - Output is an `ExtractionResult(doc_id, doc_type, fields)` stored in `state["extractions"]`

7. Classification
   - `classify_node` in `graph/nodes.py`
   - Uses the extraction results and documents to determine:
     - age category
     - residential status
     - regime preference
   - It fills `TaxpayerProfile`:
     - `AgeCategory.BELOW_60`, `SENIOR`, `SUPER_SENIOR`
     - `residential_status` = `resident` or `non_resident`
     - `regime_preference` = `"auto"`, `"old"`, or `"new"`
   - This classification can also emit `ReviewItem`s if the facts are ambiguous.

8. Research / deduction proposal
   - `research_node` in `graph/nodes.py`
   - It integrates with:
     - `src/taxpilot/knowledge/retriever.py`
     - `src/taxpilot/knowledge/store.py`
     - `src/taxpilot/knowledge/corpus_loader.py`
   - `get_retriever()` builds a BM25 retriever over the corpus under `data/corpus/`
   - `Retriever.retrieve_multi(queries)`:
     - runs multiple queries against the BM25 store
     - deduplicates by passage
     - ranks by relevance
     - returns `Passage` objects
   - `format_context(passages)` prints numbered context blocks with `rule_ids` and the underlying text
   - The model proposes deductions as `DeductionCandidate` entries:
     - `name`
     - `kind` (`CHAPTER_VIA`, `HOUSE_PROPERTY`, `SALARY_EXEMPTION`)
     - `amount`
     - `citation`
     - `rationale`
     - `confidence`
     - `needs_review`
     - `grounded`
   - This is where the “grounding” concept is enforced:
     - the rule must be present in the corpus
     - the citation must resolve to a known rule
     - the output guardrails later verify it

9. Deterministic tax computation
   - `calculate_node` in `graph/nodes.py`
   - Calls `src/taxpilot/calc/engine.py`
   - `compute(profile, income, deductions, tax_year=2024)`
     - sums salary, interest, other income
     - sums TDS
     - computes both old and new regimes
     - chooses:
       - forced regime if `profile.regime_preference` is set
       - otherwise the lower tax under “auto”
   - Important helpers:
     - `slab_tax()`
     - `_slabs_for()`
     - `_standard_deduction()`
     - `_rebate_87a()`
     - `_surcharge()`
     - `_chapter_via()`
     - `_house_property_loss()`
   - `TaxReturn` is built by `_build_return(...)`
   - Every tax line includes `TaxLine` objects with:
     - `line`
     - `label`
     - `amount`
     - `citation`
   - Rule IDs come from `src/taxpilot/knowledge/rules.py` via `cite(rule_id)`

10. Summary phase
   - `summarize_node` in `graph/nodes.py`
   - Creates a human-readable summary of the computed return and the major deductions.
   - This summary is useful both:
     - while a review is pending
     - after resuming and recomputing

11. Audit scoring
   - `audit_node` in `graph/nodes.py`
   - Calls `src/taxpilot/audit/scorer.py`
   - `score_return(tax_return, income, deductions, extractions)`:
     - checks TDS vs Form 26AS
     - checks deductions at or near cap
     - checks large deduction ratios
     - checks large donation claims
     - checks review-needed deductions
     - checks low-confidence extractions
     - checks large refund relative to income
   - Returns `AuditRisk(score, band, flags)` with weighted red flags.
   - The audit score is a heuristic, not a trained model; it is transparent and explainable.

12. Review gate
   - `review_gate_node` in `graph/nodes.py`
   - This is the human interrupt point.
   - The gate raises `interrupt()` when:
     - risk is above threshold
     - refund or payable exceeds threshold
     - confidence is low
     - deductions need confirmation
     - the return is out of scope for the deterministic engine path
   - `pipeline.pending_review(thread_id)` fetches the review payload.
   - This is the “pause and wait for human approval” loop.

13. Human correction flow
   - Client calls `GET /pending/{thread_id}` to inspect reasons.
   - Client then calls `POST /resume` with `Correction` objects.
   - `pipeline.resume()`:
     - gets graph state for the thread
     - sends `Command(resume={"corrections": [...]})`
   - Builder routes via `after_review()`:
     - if `recompute` was triggered, returns to `calculate`
     - otherwise goes to `report`

14. Report generation
   - `report_node` in `graph/nodes.py`
   - Uses the computed `TaxReturn`, citations, and deductions to produce plain-language narrative.
   - It writes `state["report"]`.
   - The repo explicitly enforces the rule: “The LLM reads, retrieves, explains, and spots anomalies; the deterministic engine computes the numbers.”

15. Guardrail output
   - `guardrail_output_node` in `graph/nodes.py`
   - Calls `src/taxpilot/guardrails/pipeline.py`
   - `Guardrails.check_output(report, tax_return, deductions, use_llm=True)`
   - Layer sequence:
     - `shape`
     - `injection`
     - `pii_inventory`
     - `figure_integrity`
     - `citation_grounding`
     - `pii_redaction`
     - `disclaimer`
     - optional `output_review`
   - The key invariant is `figure_integrity`:
     - all rupee figures in the narrative must be drawn from:
       - engine output
       - published constants
       - grounded deduction values
     - if the report invents any `₹` values, it is rejected.
   - PII redaction masks PAN/Aadhaar for display.
   - A disclaimer is appended to mark it as a draft return.

16. Finalization
   - `finalize_node` in `graph/nodes.py`
   - It bundles the state into final structured output.
   - `pipeline._to_response(...)` packages it into `ReturnResponse`.
   - Response fields include:
     - `tax_year`
     - `regime`
     - `tax_return`
     - `audit`
     - `report`
     - `llm_summary`
     - `used_llm`
     - `citations`
     - `awaiting_review`
     - `review_items`
     - `thread_id`
     - `guardrails`
     - `blocked`
     - `block_message`
     - `trace`
     - token counts
     - latency

Database and LLM interaction map

- No relational database is used in the app code path.
- The only “state persistence” is:
  - in-memory LangGraph checkpointer (`MemorySaver`) for the paused review thread
  - BM25 in-memory knowledge store for the corpus
- The actual external ML interaction is through:
  - `src/taxpilot/llm/client.py`
  - `LLMClient.complete()`
  - `LLMClient.complete_json()`
  - provider switch:
    - Anthropic via `anthropic.Anthropic(...)`
    - or Gemini via `get_llm()` selecting the Gemini client when configured
- Structured model outputs are enforced by JSON-schema validation, so the extractor, classifier, researcher, and guardrails all expect strongly typed data.
- If `use_llm=False` or no API key is configured, the system runs on deterministic paths only. The README explicitly says the suite and pipeline still work offline.

What connects each component

- `api/main.py` -> `graph/pipeline.py` -> `graph/builder.py` -> `graph/nodes.py`
- `graph/nodes.py` -> `extraction/extractor.py` -> `models.py` -> `TaxState`
- `graph/nodes.py` -> `knowledge/retriever.py` -> `knowledge/store.py` -> `data/corpus`
- `graph/nodes.py` -> `calc/engine.py` -> `knowledge/rules.py` -> engine outputs
- `graph/nodes.py` -> `audit/scorer.py` -> review gate
- `graph/nodes.py` -> `guardrails/pipeline.py` -> final `ReturnResponse`
- `api/main.py` returns the response to the UI or caller

The exact final output contract is a `ReturnResponse` object, not a database row. The UI reads it and shows:
- summary
- report
- sources
- audit
- guardrails
- trace

The final artifact is an explainable draft tax return: numbers come from the deterministic engine, explanations and citations come from the retriever and LLM, and the output is checked by guardrails before being returned.