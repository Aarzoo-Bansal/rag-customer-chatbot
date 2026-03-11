# Customer Support RAG Chatbot

A conversational AI chatbot that answers customer support questions by retrieving relevant information from policy documents. Built with a 3-node LangGraph pipeline using FAISS (Meta's similarity search library) for vector retrieval and Llama 3.1 (Meta's open-source LLM) for generation — fully local, no API keys required.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.11-red)

## Demo

![Customer Support Chatbot Demo](https://github.com/Aarzoo-Bansal/rag-customer-chatbot/blob/main/ChatBot_GIF.gif)

---

## Architecture

The pipeline is a 3-node stateful graph built with LangGraph. Each node reads from and writes to a shared `RAGState` typed dictionary, and edges define a fixed execution order.

```
User Question
     │
     ▼
┌─────────────┐
│ Query       │  ← LLM rewrites follow-ups into standalone questions
│ Rewrite     │    (skipped if no chat history — avoids unnecessary LLM calls)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ FAISS       │  ← Embedding model converts query to 384-dim vector,
│ Retrieval   │    FAISS returns top-5 similar chunks via cosine similarity
└─────┬───────┘
      │
      ▼
┌──────────────────────────┐
│ Generate Answer (LLM)    │  ← Produces grounded response using retrieved
│                          │    context + sliding window chat history (10 turns)
└──────────────────────────┘
```

### Design Decisions

**Why LangGraph over a simple chain?** LangGraph's `StateGraph` gives each node access to a shared typed state, which makes it straightforward to add new nodes (evaluation, reranking) without restructuring the pipeline. A LangChain chain would require refactoring to support conditional routing or branching.

**Why FAISS over Pinecone/Chroma?** FAISS runs locally with zero infrastructure overhead, supports exact and approximate nearest-neighbor search, and handles the dataset size (44 chunks) with sub-millisecond retrieval. For this scale, a managed vector database would add latency and complexity without benefit.

**Why Llama 3.1 8B via Ollama?** Keeps the entire system local and reproducible — no API keys, no rate limits, no cost per query. Temperature is set to 0 for deterministic outputs, which is essential for customer support where consistent answers matter.

**Why 300-character chunks with 40-character overlap?** Tested against 500 and 800-character alternatives. Smaller chunks produced more precise retrieval for the policy documents (which contain short, self-contained policy statements), though they occasionally return sentence fragments. The overlap prevents hard cuts at sentence boundaries.

---

## Key Engineering Patterns

**Query Rewriting with Fallback** — Follow-up questions ("what about exchanges?") are rewritten into standalone queries using chat history context. Small models sometimes answer instead of rewriting, so a validation layer checks for empty, oversized (>500 chars), or multi-line rewrites and falls back to the raw question. This prevents retrieval from being poisoned by a bad rewrite.

**Sliding Window History** — Chat history is capped at the last 10 turns (20 messages) to keep inference fast and prevent context window overflow. The full conversation remains visible in the UI — only the LLM-facing history is trimmed.

**Cached Resources** — The embedding model, FAISS index, LLM connection, and compiled graph are loaded once using `@st.cache_resource`. Streamlit reruns the entire script on every interaction, so without caching, every message would reload the 77MB embedding model and reconnect to Ollama.

**Startup Health Checks** — The app verifies both the FAISS index and Ollama connection at startup. If either is unavailable, it shows a user-friendly warning instead of crashing mid-conversation.

**Source Attribution** — Every response includes expandable source documents with file-level provenance, so the user can verify claims against the original policy text.

---

## Known Tradeoffs & Limitations

**Dense-only retrieval misses keyword matches.** FAISS retrieves by semantic similarity, which means exact-term queries ("30-day return policy") can rank lower than semantically similar but less precise chunks. Hybrid retrieval (FAISS + BM25) would address this by combining semantic and keyword search.

**No answer validation.** The pipeline trusts the LLM's output. If the retrieved context is noisy or the model hallucinates, the user sees an incorrect answer with no warning. An evaluation node that checks faithfulness (is the answer grounded in context?) before returning would catch this.

**Small models struggle with multi-step reasoning.** Llama 8B correctly states policy rules but sometimes applies them incorrectly to specific scenarios (e.g., date-comparison questions). Explicit logic guidelines in the system prompt help but don't fully solve this — it's a fundamental limitation of the model size.

**PDF extraction produces messy text.** PyPDFLoader outputs raw text with double spaces, broken lines, and lost table formatting. This corrupts chunks and degrades both retrieval quality and generation accuracy. Structured extraction (Unstructured.io) or vision-based approaches (ColPali) would preserve document layout.

---

## Production Roadmap

If extending this system for production use, these are the changes I'd prioritize, in order:

**1. Hybrid Retrieval (FAISS + BM25)** — Add BM25 sparse retrieval alongside FAISS dense retrieval using LangChain's `EnsembleRetriever`. Weight semantic search at 60%, keyword search at 40%. Deduplicate results from both retrievers. This improves recall for exact-term queries without sacrificing semantic understanding.

**2. Cross-Encoder Reranking** — After hybrid retrieval, score each document's relevance using a dedicated cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22MB). This is orders of magnitude faster than LLM-based reranking and trained specifically for the relevance scoring task. Keep top 3 from the reranked list to reduce noise in the LLM's context.

**3. Answer Evaluation with Hallucination Detection** — Add an evaluation node after generation that checks faithfulness (is the answer supported by retrieved context?), completeness (does it address the question?), and confidence (1-10 score). Use the LLM-as-Judge pattern initially, migrate to RAGAS framework (faithfulness, answer relevancy, context precision metrics) for quantitative evaluation.

**4. Conditional Routing & Human-in-the-Loop** — Replace the linear graph with conditional edges. After evaluation, route based on confidence: high confidence → return answer, low confidence or hallucination detected → escalate with a disclaimer directing the user to human support. This is the architectural shift from a linear pipeline to an agentic workflow.

**5. Tiered Safety by Domain** — For multi-domain knowledge bases (e.g., university departments), assign risk levels to document sources. Low-risk topics (IT, general info) tolerate medium-confidence answers. High-risk topics (immigration, legal, financial) require strict confidence thresholds and always include verification disclaimers and source links.

**6. Data Freshness Pipeline** — Implement scheduled re-scraping with content hashing for change detection, diff analysis for identifying what changed, temporal metadata for date-sensitive content, and version history for audit trails. Flag stale content rather than serving outdated answers.

**7. Evaluation Dataset & CI/CD** — Build a test suite of 50+ question-answer pairs with known correct answers. Run the pipeline against this dataset automatically on every code change. Block deploys if accuracy drops below threshold. This is the AI equivalent of unit tests.

---

## Project Structure

```
.
├── rag_chatbot.ipynb         # Step-by-step RAG pipeline walkthrough
├── app.py                    # Streamlit chatbot with LangGraph
├── data/
│   ├── Everstorm_Return_and_exchange_policy.pdf
│   ├── Everstorm_Shipping_and_Delivery_Policy.pdf
│   ├── Everstorm_Payment_refund_and_security.pdf
│   └── Everstorm_Product_sizing_and_care_guide.pdf
├── faiss_index/
│   ├── index.faiss           # Vector embeddings
│   └── index.pkl             # Document store + ID mappings
├── environment.yml           # Conda dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11
- Conda or Miniconda
- [Ollama](https://ollama.com/download) installed

### 1. Clone and create environment

```bash
git clone <repo-url>
cd week2-rag
conda env create -f environment.yml
conda activate rag-chatbot
```

### 2. Start Ollama and pull the model

```bash
ollama serve
```

In a new terminal:

```bash
ollama pull llama3.1:8b
```

> **Note:** The 8B model requires ~5GB disk and ~6GB RAM. For lower-resource machines, use `gemma3:4b` or `gemma3:1b` and update the model name in `app.py`.

### 3. Build the FAISS index

Run `rag_chatbot.ipynb` top to bottom. This loads PDFs, chunks text, generates embeddings with `gte-small`, and saves the FAISS index to `faiss_index/`.

### 4. Run the chatbot

```bash
streamlit run app.py
```

Open http://localhost:8501.

---

## Customization

**Use a different LLM:**

```bash
ollama pull <model-name>
```

Update `load_llm()` in `app.py`:

```python
def load_llm():
    return ChatOllama(model="<model-name>", temperature=0)
```

**Use a different embedding model:**

Update both the notebook and `load_vectordb()` in `app.py`:

```python
emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

> **Important:** Changing the embedding model requires rebuilding the FAISS index by rerunning the notebook.

**Add your own documents:**

Place PDFs in `data/`, update the glob pattern in `load_offline_files()`, and rebuild the index.

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | HuggingFace `gte-small` | 384-dim vectors, runs locally, no API keys |
| Vector Store | FAISS | Meta's similarity search library — sub-millisecond retrieval |
| LLM | Llama 3.1 8B via Ollama | Meta's open-source model — local, deterministic, free |
| Orchestration | LangGraph | Stateful graph with typed state — extensible to conditional routing |
| Framework | LangChain | Document loading, text splitting, retriever abstraction |
| UI | Streamlit | Chat interface with session state and resource caching |