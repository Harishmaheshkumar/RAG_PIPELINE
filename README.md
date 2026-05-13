<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Self-Improving%20RAG%20Pipeline&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Production-grade%20Retrieval-Augmented%20Generation%20with%20Hallucination%20Detection&descAlignY=58&descSize=16" alt="Banner"/>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Gemini-Powered-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-blue?style=flat-square"/>
</p>

<br/>

> **🧠 Ask questions. Get grounded answers. Catch hallucinations — automatically.**  
> A full-stack RAG system that retrieves, reranks, generates, and *evaluates itself.*

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📡 API Reference](#-api-reference) · [⚙️ Configuration](#️-configuration) · [📊 MLflow Tracking](#-mlflow-tracking)

</div>

---

## ✨ What Makes This Special

Unlike a basic RAG setup, this pipeline **closes the loop** — it doesn't just answer your question, it audits its own answer for hallucinations using a second LLM call and logs every run to MLflow for continuous improvement.

```
📄 Your Documents  →  🔍 Semantic Search  →  🔀 Reranking  →  🤖 Answer  →  🧪 Self-Evaluation  →  📊 MLflow Log
```

<table>
<tr>
<td>

### 🔍 Smart Retrieval
Cosine-similarity vector search over ChromaDB with persistent storage — your knowledge base survives restarts.

</td>
<td>

### 🔀 Semantic Reranking
A second-pass reranker computes query-document cosine similarity to bubble up the *most relevant* chunks.

</td>
</tr>
<tr>
<td>

### 🧪 Hallucination Detection
An LLM-as-judge evaluator scores every answer (0.0 → 1.0) for factual consistency against retrieved sources.

</td>
<td>

### 📊 Experiment Tracking
Every query, retrieval count, and hallucination score is logged to MLflow — build datasets, tune parameters.

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                                                                  │
│  POST /ingest-files    POST /query         GET /health           │
└───────────┬────────────────┬──────────────────────┬─────────────┘
            │                │                      │
            ▼                ▼                      │
   ┌──────────────┐  ┌───────────────────┐          │
   │ FileProcessor│  │   RAG Pipeline    │    ┌─────┴──────┐
   │              │  │                   │    │   Health   │
   │  .pdf  .docx │  │  1. Retrieve      │    │   Check    │
   │  .txt  .json │  │  2. Rerank        │    └────────────┘
   │  .md   .py   │  │  3. Generate      │
   └──────┬───────┘  │  4. Evaluate      │
          │          └─────┬─────────────┘
          ▼                │
   ┌──────────────┐        │
   │ ChromaDB     │◄───────┤
   │ Vector Store │        │
   │              │        ▼
   │  Gemini EF   │  ┌──────────────┐    ┌──────────────┐
   │  or ST-EF    │  │  Gemini LLM  │    │   MLflow     │
   └──────────────┘  │  (Generate + │    │   Tracking   │
                     │   Evaluate)  │    └──────────────┘
                     └──────────────┘
```

### 📦 Module Overview

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app, routing, file ingestion endpoint |
| `pipeline.py` | Orchestrates retrieve → rerank → generate → evaluate |
| `embedding_provider.py` | Gemini `text-embedding-004` or local SentenceTransformer |
| `vector_store.py` | ChromaDB persistent collection CRUD |
| `reranker.py` | Vectorized cosine-similarity reranking |
| `evaluator.py` | LLM-as-judge hallucination scoring |
| `mlflow_logger.py` | Thread-safe MLflow experiment logging |
| `file_processor.py` | Multi-format text extraction (PDF, DOCX, TXT…) |
| `config.py` | Pydantic settings with `.env` support |
| `schemas.py` | Request/response Pydantic models |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

```env
# .env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional overrides (defaults shown)
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
CHROMA_PERSIST_DIR=./chroma_store
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=rag_pipeline
MAX_RETRIEVAL_RESULTS=10
RERANK_TOP_K=5
RESPONSE_TEMPERATURE=0.2
```

> 💡 **No Gemini key?** The pipeline automatically falls back to `SentenceTransformer (all-MiniLM-L6-v2)` for local embeddings.

### 3. Launch

```bash
# Start the API server
uvicorn app.main:app --reload --port 8000

# (Optional) Start MLflow UI in a separate terminal
mlflow ui --port 5000
```

🟢 API live at `http://localhost:8000`  
📊 MLflow UI at `http://localhost:5000`

---

## 📡 API Reference

### `POST /ingest-files` — Upload Documents

Upload one or more files to populate the knowledge base.

```bash
curl -X POST http://localhost:8000/ingest-files \
  -F "files=@research_paper.pdf" \
  -F "files=@notes.txt"
```

```json
{
  "status": "success",
  "ingested_files_count": 2,
  "ingested_ids": ["research_paper.pdf_4821", "notes.txt_9034"]
}
```

**Supported formats:** `.pdf` · `.docx` · `.doc` · `.txt` · `.md` · `.py` · `.json`

---

### `POST /ingest` — Ingest Raw Text

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc-001",
        "text": "ChromaDB is a vector database optimized for AI applications.",
        "metadata": {"source": "manual", "topic": "databases"}
      }
    ]
  }'
```

---

### `POST /query` — Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ChromaDB used for?", "top_k": 5}'
```

<details>
<summary><b>📋 Full Response Schema</b></summary>

```json
{
  "query": "What is ChromaDB used for?",
  "answer": "ChromaDB is a vector database optimized for AI applications...",
  "hallucination_score": 0.05,
  "consistency": "high",
  "evaluation_rationale": "The answer is fully grounded in Source 1 with no external claims.",
  "sources": [
    {
      "id": "doc-001",
      "text": "ChromaDB is a vector database...",
      "metadata": {"source": "manual"},
      "score": 0.12
    }
  ],
  "reranked_sources": [
    {
      "id": "doc-001",
      "text": "ChromaDB is a vector database...",
      "metadata": {"source": "manual"},
      "score": 0.12,
      "rerank_score": 0.94
    }
  ]
}
```

</details>

---

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "pipeline_ready": true}
```

---

## ⚙️ Configuration

All settings are managed via Pydantic `BaseSettings` and can be set in `.env` or as environment variables.

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | `None` | Gemini API key (falls back to local embeddings if absent) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model for generation and evaluation |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_store` | Path for ChromaDB persistence |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT` | `rag_pipeline` | MLflow experiment name |
| `MAX_RETRIEVAL_RESULTS` | `10` | Documents fetched from vector store |
| `RERANK_TOP_K` | `5` | Top-K kept after reranking |
| `RESPONSE_TEMPERATURE` | `0.2` | Generation temperature (lower = more factual) |

---

## 📊 MLflow Tracking

Every `/query` call automatically logs:

| Logged Item | Type | Example |
|---|---|---|
| `query_text` | Parameter | `"What is RAG?"` |
| `retrieved_count` | Metric | `10` |
| `reranked_count` | Metric | `5` |
| `hallucination_score` | Metric | `0.07` |
| `evaluation_details.json` | Artifact | Full eval payload |

```bash
# Launch the MLflow dashboard
mlflow ui --port 5000
```

> Track hallucination trends over time, compare temperature settings, and build a ground-truth eval dataset — all from the MLflow UI.

---

## 🔌 Embedding Providers

The pipeline auto-selects the best available provider:

```
GOOGLE_API_KEY present?
    YES → GeminiEmbeddingProvider  (text-embedding-004, cloud)
    NO  → SentenceTransformerEmbeddingProvider  (all-MiniLM-L6-v2, local)
```

Both implement the `BaseEmbeddingProvider` interface, so you can plug in any custom provider:

```python
class MyCustomProvider(BaseEmbeddingProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...  # your logic

    def chroma_embedding_function(self):
        ...  # ChromaDB-compatible wrapper
```

---

## 🧪 Hallucination Scoring

The `ResponseEvaluator` sends the query, generated answer, and retrieved sources to Gemini and asks it to return structured JSON:

```json
{
  "hallucination_score": 0.05,   // 0.0 = faithful, 1.0 = hallucinated
  "consistency": "high",          // high / medium / low
  "rationale": "Answer is fully grounded in Source 1..."
}
```

The evaluator uses `response_mime_type: "application/json"` for reliable structured output and falls back gracefully on parse errors.

---

## 📁 Project Structure

```
rag-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── pipeline.py          # Core RAG orchestration
│   ├── config.py            # Pydantic settings
│   ├── schemas.py           # Request/response models
│   ├── embedding_provider.py
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── reranker.py          # Semantic reranker
│   ├── evaluator.py         # Hallucination evaluator
│   ├── file_processor.py    # Multi-format text extraction
│   ├── mlflow_logger.py     # Thread-safe MLflow logging
│   └── static/              # Optional frontend (index.html)
├── chroma_store/            # Persisted vector data (auto-created)
├── .env                     # Your secrets (not committed)
├── .env.example
└── requirements.txt
```

---

## 🛠️ Requirements

```txt
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-dotenv
google-genai
chromadb
numpy
mlflow
pypdf
python-docx
sentence-transformers   # Optional: for local embeddings
```

---

## 🗺️ Roadmap

- [ ] 🌐 Streaming responses via SSE
- [ ] 🗂️ Multi-collection / multi-tenant support  
- [ ] 📈 Automatic prompt refinement based on hallucination trends
- [ ] 🔐 API key authentication middleware
- [ ] 🐳 Docker + Docker Compose setup
- [ ] 🧪 Pytest suite with mock embeddings

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss major changes.

```bash
git checkout -b feature/your-feature
# make your changes
git commit -m "feat: describe your change"
git push origin feature/your-feature
# open a Pull Request
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" alt="Footer"/>

**Built with ❤️ using FastAPI · ChromaDB · Google Gemini · MLflow**

⭐ Star this repo if it helped you build something cool!

</div>
