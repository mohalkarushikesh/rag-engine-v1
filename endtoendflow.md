# RAG-app — End-to-End Flow

A Retrieval-Augmented Generation app that ingests PDFs and answers questions grounded in their passages. **One codebase runs two ways**, chosen at runtime by the `RAG_BACKEND` environment variable:

- **`local`** (default) — fully offline, HuggingFace models, extractive answering. No cloud, no API keys.
- **`bedrock`** — AWS-native, Amazon Bedrock embeddings + Claude generation. Deployed on App Runner.

Everything except two swap points — the **embedding model** and the **answerer** — is shared between the two backends. PDF extraction/OCR, chunking, FAISS indexing, hybrid retrieval, caching, the Flask API, and the UI are identical regardless of backend.

---

## 1. High-Level Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Flask web app               │
                         │  app.py  (gunicorn in prod)              │
                         │  GET /  /healthz  /api/status            │
                         │  POST /api/ask                           │
                         └───────────────┬─────────────────────────┘
                                         │ builds once (daemon thread)
                                         ▼
                         ┌─────────────────────────────────────────┐
                         │          build_pipeline()                │
                         │          rag_pipeline.py                 │
                         └───────────────┬─────────────────────────┘
                                         │
        ┌────────────────────────────────┼─────────────────────────────────┐
        │ SHARED                          │ SHARED                          │
        ▼                                 ▼                                 ▼
  Extract text / OCR  ──►  Chunk (1000/200)  ──►  Embed + FAISS index  ──►  Hybrid Retriever
  (pypdf + RapidOCR)       RecursiveCharacter       (cached on disk)        (semantic + keyword)
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              │  SWAP POINT 1: load_embeddings()                        │
                              │   local   → HuggingFaceEmbeddings(distilbert)           │
                              │   bedrock → BedrockEmbeddings(titan-embed-v2)           │
                              └─────────────────────────────────────────────────────────┘
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              │  SWAP POINT 2: build_chain()  →  .invoke({"query"})     │
                              │   local   → RetrieverReaderQA   (extractive reader)     │
                              │   bedrock → BedrockGenerativeQA (Claude generates)      │
                              └─────────────────────────────────────────────────────────┘
```

Both answerers expose the **same contract** — `.invoke({"query": q}) -> {"result": ...}` — so the Flask layer and the CLI never branch on backend.

---

## 2. Component Map

| File | Role |
|---|---|
| `rag_pipeline.py` | Core pipeline + CLI: extraction, OCR, chunking, embeddings, FAISS, hybrid retrieval, both answerers, backend selection. |
| `app.py` | Flask server; builds the pipeline once in a background thread, serves UI + JSON API. |
| `templates/index.html` | Single-page UI (status chip, question box, answer card). |
| `static/js/app.js` | Polls `/api/status`, POSTs to `/api/ask`. |
| `static/css/styles.css` | USWDS-styled layout. |
| `.rag_cache/` | On-disk caches: extracted text (`text_<hash>.json`) + FAISS index (`<signature>/index.faiss`, `index.pkl`). |
| `scripts/download_models.py` | Bakes local HF models into the Docker image at build time. |
| `Dockerfile` | CPU-only image; offline models; gunicorn entrypoint. |
| `infra/` | Terraform: ECR + App Runner + IAM (Bedrock least-privilege). |
| `deploy.sh` | Build → push ECR → `terraform apply`. |
| `.github/workflows/deploy.yml` | CI "Deploy RAG app to AWS". |
| `.env.example`, `requirements.txt` | Runtime config + dependencies. |

---

## 3. Shared Pipeline (both backends)

Orchestrated by `build_pipeline()` in `rag_pipeline.py`:

1. **Extract text** — `extract_texts()`. Fast path: `pypdf.PdfReader.extract_text()`. If a page/file yields no text (scanned or outlined PDF), falls back to **OCR** via `ocr_pdf()`: PyMuPDF renders each page, `RapidOCR` (offline ONNX) reads it.
2. **Cache extracted text** — `get_texts()` writes `.rag_cache/text_<hash>.json`, keyed on **PDF content hash** (`_files_signature`), so OCR runs exactly once and the sidecar is portable/commitable.
3. **Chunk** — `split_into_chunks()`: each page becomes a LangChain `Document`, then `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
4. **Embed + index** — `build_vector_store()`: `FAISS.from_documents(documents, embedding_model)`.
5. **Cache the FAISS index** — saved to `.rag_cache/<signature>/`; reloaded with `FAISS.load_local(...)`. The signature (`_source_signature`) includes the **effective embedding id** + chunk settings (`emb=...;chunk=1000/200`), so **local and Bedrock indexes never collide** — switching backends triggers a clean reindex.

### Hybrid retrieval (shared) — `HybridRetriever`

- **Semantic arm:** `vector_store.similarity_search(query, k=SEMANTIC_K)` over FAISS.
- **Keyword arm:** content-word overlap — `_keywords()` lowercases/tokenizes, drops stopwords and 1-char tokens, scores chunks by keyword-intersection size. Added because weak local embeddings miss exact-term questions (e.g. "What is Acrophobia?").
- `contexts()` unions both arms, de-duped by passage text.
- Knobs: `RAG_SEMANTIC_K` (default 5), `RAG_KEYWORD_K` (default 5).

---

## 4. Backend Selection

Read once at import in `rag_pipeline.py`:

```python
RAG_BACKEND = os.environ.get("RAG_BACKEND", "local").strip().lower()
```

Only two functions branch on it:

| Swap point | `local` | `bedrock` |
|---|---|---|
| **Embeddings** — `load_embeddings()` | `HuggingFaceEmbeddings(model_name="distilbert-base-uncased")` | `BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, region_name=BEDROCK_REGION)` (lazy `langchain_aws` import) |
| **Answerer** — `build_chain()` | `RetrieverReaderQA`: hybrid retriever + `ExtractiveReader` (`deepset/roberta-base-squad2`), scores all passages in one batched forward pass, returns best answer span | `BedrockGenerativeQA`: hybrid retriever + `ChatBedrock`, stuffs passages into a grounded prompt, generates with `temperature=0, max_tokens=512` |

`clean()` also branches: local keeps the first paragraph; bedrock returns the full generated answer. The lazy AWS imports mean the offline path never needs `boto3`/`langchain-aws` installed at runtime.

### Models

| | Local | Bedrock |
|---|---|---|
| Embeddings | `distilbert-base-uncased` (768-dim, mean-pooled) | `amazon.titan-embed-text-v2:0` |
| Answerer | `deepset/roberta-base-squad2` (extractive QA) | `anthropic.claude-3-haiku-20240307-v1:0` (generative) |
| OCR | RapidOCR (ONNX, offline) | RapidOCR (same) |

---

## 5. Request Flow (runtime)

### Startup

`app.py` builds the pipeline **once** in a daemon thread via `start_build_once()` — triggered at module import so it runs under gunicorn too. State lives in a module-level `_state` dict (`status: loading|ready|error`, `message`, `backend`) guarded by `_lock`.

### API

| Route | Method | Behavior |
|---|---|---|
| `/` | GET | Renders `index.html` with configured `pdf_paths`. |
| `/healthz` | GET | Liveness probe; `{"ok": true}` 200 as soon as the web process is up (decoupled from index build — used by App Runner health check). |
| `/api/status` | GET | `{status, message, backend}`. UI polls this every 1.5s. |
| `/api/ask` | POST | Body `{"query": "..."}` → `{"answer": "..."}`. 503 if not ready, 400 if empty. |

### Ask path (identical for both backends)

```
UI textarea → POST /api/ask {query}
  → chain.invoke({"query": q})
      → HybridRetriever.contexts(q)         # semantic + keyword, de-duped
      → local:   ExtractiveReader picks best span across passages
        bedrock: ChatBedrock generates grounded answer from passages
      → clean(result)
  → {"answer": ...} → rendered in answer card
```

---

## 6. Running Locally (`RAG_BACKEND=local`, default)

```bash
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5000
```

Custom documents:

```bash
RAG_PDF_PATHS="a.pdf,b.pdf" python app.py
```

- Runs **fully offline** — `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set at the top of `rag_pipeline.py` before transformers imports.
- First build does extract → (OCR if needed) → chunk → embed → index. Subsequent startups reload the cache (~8s).
- CLI alternatives:
  ```bash
  python rag_pipeline.py "your question"   # one-shot
  python rag_pipeline.py                   # interactive REPL
  python rag_pipeline.py extract           # run OCR once, populate text cache
  ```

### Testing the Bedrock path locally

Needs AWS credentials (standard chain: `aws configure` / `AWS_PROFILE` / env keys) and Bedrock model access in the region:

```bash
RAG_BACKEND=bedrock AWS_REGION=us-east-1 python app.py
```

This reindexes with Titan embeddings (new cache signature) and answers via Claude Haiku.

---

## 7. Deploying to AWS (`RAG_BACKEND=bedrock`)

### One command

```bash
AWS_REGION=us-east-1 IMAGE_TAG=$(git rev-parse --short HEAD) ./deploy.sh
```

`deploy.sh` runs:

1. `terraform init` + targeted apply of the **ECR repo**.
2. ECR docker login.
3. Build image (bakes local HF models via `download_models.py` — needs internet, so run from CI/CodeBuild or an off-corp-network box), tag, push.
4. `terraform apply` the **App Runner** service.
5. Print `service_url` (public HTTPS).

### What Terraform provisions (`infra/`)

- **ECR** repo — scan-on-push, lifecycle keep-last-10, `force_delete`.
- **Two IAM roles:**
  - *Access role* (`build.apprunner.amazonaws.com`) — ECR pull.
  - *Instance role* (`tasks.apprunner.amazonaws.com`) — **least-privilege Bedrock**: `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream` on `arn:aws:bedrock:*::foundation-model/*`. **No API keys stored** — the app calls Bedrock via this role.
- **App Runner service** — injects env vars (`RAG_BACKEND=bedrock`, `AWS_REGION`, `RAG_BEDROCK_EMBED_MODEL`, `RAG_BEDROCK_CHAT_MODEL`, `RAG_PDF_PATHS`, `PORT=8080`), `/healthz` health check, autoscaling (min 1 / max 3 / max-concurrency 50), 1024 CPU / 3072 MB. Bedrock is a public API — no VPC connector needed.
- Terraform defaults set `rag_backend=bedrock` for AWS (vs `local` default in code).

### CI alternative

The **"Deploy RAG app to AWS"** GitHub workflow (`.github/workflows/deploy.yml`) runs the same `deploy.sh` on manual dispatch or push to `main` (app/infra paths). Auth via OIDC (`AWS_ROLE_ARN`) or access-key secrets fallback; `IMAGE_TAG` = input or commit SHA.

### Teardown

```bash
terraform -chdir=infra destroy
```

---

## 8. Environment Variables

| Var | Default | Applies to | Purpose |
|---|---|---|---|
| `RAG_BACKEND` | `local` | both | `local` or `bedrock`. |
| `RAG_PDF_PATHS` | sample PDF | both | Comma-separated PDFs to ingest. |
| `RAG_SEMANTIC_K` | `5` | both | Semantic retrieval depth. |
| `RAG_KEYWORD_K` | `5` | both | Keyword-arm depth. |
| `RAG_CACHE_DIR` | `.rag_cache` | both | Cache location. |
| `RAG_QA_MODEL` | `deepset/roberta-base-squad2` | local | Extractive reader model. |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | `us-east-1` | bedrock | Bedrock region. |
| `RAG_BEDROCK_EMBED_MODEL` | `amazon.titan-embed-text-v2:0` | bedrock | Embedding model id. |
| `RAG_BEDROCK_CHAT_MODEL` | `anthropic.claude-3-haiku-20240307-v1:0` | bedrock | Chat model id. |
| `HOST` / `PORT` | `127.0.0.1` / `5000` (`8080` in container) | both | Bind address. |

---

## 9. Backend Comparison at a Glance

| Aspect | `local` | `bedrock` |
|---|---|---|
| Network | Fully offline | Calls AWS Bedrock |
| Auth | None | IAM instance role (no keys) |
| Embeddings | distilbert (HF) | Titan Embed v2 |
| Answering | Extractive span (RoBERTa) | Generative (Claude Haiku) |
| Answer style | Verbatim span from passage | Fluent grounded generation |
| Cost | Compute only | Per-token Bedrock usage |
| Deploy target | Laptop / any box | App Runner (HTTPS, autoscale) |
| Cache signature | `emb=distilbert;...` | `emb=titan;...` (separate index) |

---

*Shared design insight: retrieval, extraction/OCR, chunking, and caching are backend-agnostic. Only `load_embeddings()` and `build_chain()` swap, both behind a common `.invoke({"query"}) → {"result"}` interface — so the API, UI, and CLI are written once and run unchanged on either backend.*
