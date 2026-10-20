# End-to-End RAG App: PDFs → Answers

A small, self-contained **Retrieval-Augmented Generation (RAG)** app. You give it PDFs, it finds the
passages relevant to your question and answers *from those passages*.

Originally prototyped in a Jupyter notebook ([RAG.ipynb](RAG.ipynb)), it now ships as a reusable
Python module plus a web application:

```
PDF -> text (cached, +OCR fallback) -> chunks -> embeddings -> FAISS
                                                                  |
                                    hybrid retrieve (semantic + keyword)
                                                                  |
                                       extractive reader -> answer -> Web UI
```

The whole thing runs **fully offline** on locally cached models — no HuggingFace Hub, no API keys,
no network calls.

---

## What it does

1. **Extract** text from PDFs (with an OCR fallback for scanned / text-as-outline PDFs). The
   extracted text is **cached**, so OCR runs only once per file (see [Caching](#caching--why-startup-is-fast)).
2. **Chunk** the text and wrap it as LangChain `Document` objects.
3. **Embed** each chunk into a vector with a local embedding model and **index** them in FAISS.
4. **Retrieve** candidate passages with a **hybrid retriever**: semantic similarity (FAISS) *and*
   keyword overlap. The keyword arm catches exact-term questions the weak embeddings miss.
5. **Read** the answer: an *extractive* QA model finds the best answer span inside the retrieved
   passages (scored in one batched pass). This is far faster on CPU than generating text.
6. **Serve** it through a web application with a clean, accessible interface.

---

## Performance

Measured on this repo's sample (`gk_ques_ans.pdf`, a 40-page scanned PDF), CPU-only (no GPU),
before vs. after the changes in this version:

| Metric                        | Before (generative)        | After (retriever + reader) |
|-------------------------------|----------------------------|----------------------------|
| **Per-question latency**      | ~50–75 s                   | **~2 s** (warm)            |
| **App startup (warm)**        | ~4–9 min (OCR every start) | **~8 s** (OCR cached)      |
| **OCR runs**                  | every startup / rebuild    | **once per file, ever**    |
| **Accuracy** (6-question set) | 4 / 6                      | **6 / 6**                  |

What changed, and why:

- **Generative LLM → extractive reader.** `microsoft/phi-1_5` (1.3B) generated tokens one at a time
  on CPU — the ~50–75 s wait. `deepset/roberta-base-squad2` instead *extracts* the answer span from
  the passages in a single forward pass (~2 s). (Instruction-tuned generators like
  `Qwen2.5-0.5B-Instruct` would also be fast, but their weights aren't in the local cache.)
- **Semantic-only → hybrid retrieval.** The `distilbert` embeddings are weak sentence encoders and
  missed exact-term questions ("What is Acrophobia?"). Adding a keyword arm fixed the misses (4/6 → 6/6).
- **OCR every startup → cached once.** Text extraction is now cached separately from the index, keyed
  on the PDF's content hash, so OCR never re-runs unless the file itself changes.

---

## Project structure

| File / folder                | Purpose                                                              |
|------------------------------|----------------------------------------------------------------------|
| `rag_pipeline.py`            | The RAG pipeline as a reusable module (+ a CLI). Converted from the notebook. |
| `app.py`                     | Flask web server: builds the pipeline once, serves the UI and a small JSON API. |
| `templates/index.html`       | The web page.                                                        |
| `static/css/styles.css`      | Styling — U.S. Web Design System (USWDS) color tokens, **Montserrat** font, light theme. |
| `static/js/app.js`           | Front-end: status polling + async question/answer.                   |
| `.rag_cache/`                | Cached extracted text (`text_*.json`) and FAISS index. Safe to delete. |
| `RAG.ipynb`                  | The original step-by-step notebook (kept for reference).             |
| `gk_ques_ans.pdf`            | Sample document used by default.                                     |

---

## Models used (all local / offline)

| Role        | Model                              | Notes                                                        |
|-------------|------------------------------------|--------------------------------------------------------------|
| Embeddings  | `distilbert-base-uncased`          | Mean-pooled → 768-dim vectors for the semantic retriever     |
| Reader (QA) | `deepset/roberta-base-squad2`      | Extractive: finds the answer span in the retrieved passages  |
| OCR         | RapidOCR (`rapidocr-onnxruntime`)  | Self-contained ONNX models; used only when a PDF has no text |

> **Why these?** They're the capable models already present in the local HuggingFace cache with full
> weights. Override the reader with `RAG_QA_MODEL` if you have a better QA model cached.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to run

### Web application (recommended)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** and ask questions.

- The one-time build (extract → embed → index → load the reader) runs in a background thread at
  startup. The page loads immediately and a status indicator shows **"Preparing index…"**; the
  **Get answer** button enables once it reads **"Index ready"**. With text + index cached, this is
  a few seconds.
- Index your own PDF(s) with an environment variable (comma-separated for multiple files):

  ```bash
  RAG_PDF_PATHS="my_document.pdf,another.pdf" python app.py            # macOS / Linux
  ```
  ```powershell
  $env:RAG_PDF_PATHS="my_document.pdf,another.pdf"; python app.py       # Windows PowerShell
  ```

#### API

| Endpoint        | Method | Body / Response                                              |
|-----------------|--------|--------------------------------------------------------------|
| `/api/status`   | GET    | `{"status": "loading｜ready｜error", "message": "..."}`       |
| `/api/ask`      | POST   | Request `{"query": "..."}` → Response `{"answer": "..."}`     |

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
     -H "Content-Type: application/json" -d '{"query":"What is Acrophobia?"}'
# {"answer":"Fear of Heights"}
```

### Command line

```bash
python rag_pipeline.py "Who is the founder of Vaccinology?"   # one-shot
python rag_pipeline.py                                        # interactive REPL
python rag_pipeline.py extract                                # run OCR once & cache the text
```

---

## Caching — why startup is fast

Two independent on-disk caches under `.rag_cache/` remove the repeated cost:

1. **Extracted text** (`text_<hash>.json`) — keyed on the **PDF's content hash**. OCR of a scanned
   PDF is the slowest step, and the pages never change unless the file does, so this runs **exactly
   once**. Re-chunking, changing the embedding model, or clearing the index will **not** re-run OCR.
   Because the key is content (not mtime), the sidecar is **portable**: run `python rag_pipeline.py
   extract` once and **commit** `.rag_cache/text_*.json`, and OCR won't run on any other machine or CI.
2. **FAISS index** — keyed on the files *and* the embedding/chunk settings, since changing those
   genuinely requires re-embedding (but never re-OCR).

Override the cache location with `RAG_CACHE_DIR`. Delete `.rag_cache/` to rebuild from scratch.

---

## OCR fallback

Some PDFs contain **no real text** — they're scanned images, or the text was converted to
outlines/curves. `pypdf` extracts nothing from those.

1. **Fast path** — `pypdf.extract_text()` for normal, text-based PDFs (instant).
2. **OCR fallback** — if a file yields no text, each page is rendered to an image with PyMuPDF and
   read with RapidOCR. Slow (a few minutes on CPU) but only ever runs once thanks to the text cache.

---

## Offline notes

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — load only cached models, set at the top of
  `rag_pipeline.py`, *before* `transformers` is imported.
- RapidOCR ships its models inside the wheel, so OCR works without any download.
- The CSS is vendored locally; only the Montserrat webfont is fetched when online (with a fallback).

---

## Tuning & limitations

- **Retrieval knobs:** `RAG_SEMANTIC_K` (default 5) and `RAG_KEYWORD_K` (default 5) control how many
  candidate chunks each arm of the hybrid retriever contributes. Higher = better recall but more
  reader compute; too high pulls in distractors.
- **Extractive answers** are spans copied verbatim from the document — ideal for factual lookups,
  not for summaries or multi-fact synthesis.
- **Model overrides:** `RAG_QA_MODEL` swaps the reader. GPU is used automatically if available
  (`torch.cuda.is_available()`); otherwise CPU with all cores.

---

## Roadmap

See [Todo.md](Todo.md) — next up: multi-document retrieval and live source retrieval (showing which
passages an answer came from). The per-file text cache already makes adding a document OCR only that
new file.


## Application 

![RAG Web Application](/assets/rag-application.png)