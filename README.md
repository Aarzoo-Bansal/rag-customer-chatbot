# Customer Support RAG Chatbot

A hands-on project for learning how to build Retrieval-Augmented Generation (RAG) pipelines from scratch. It starts with a basic RAG implementation in a Jupyter notebook and evolves into a production-aware Streamlit chatbot with spell correction, query rewriting, and code-based reasoning.

Built with LangChain, LangGraph, FAISS, and Ollama — everything runs locally, no API keys needed.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.11-red)

---

## What You'll Learn

This project walks through building a RAG system step by step:

1. **Document ingestion** — loading PDFs and web pages into a unified document format
2. **Text chunking** — splitting documents into overlapping chunks for embedding
3. **Vector embeddings** — converting text to vectors using HuggingFace models
4. **FAISS indexing** — building and querying a similarity search index
5. **LLM integration** — connecting to local models via Ollama
6. **RAG pipeline** — combining retrieval and generation into a working chatbot
7. **Conversation memory** — handling multi-turn follow-up questions
8. **LangGraph workflows** — building stateful, graph-based pipelines with conditional routing
9. **Streamlit UI** — turning the pipeline into an interactive web app
10. **Production patterns** — spell correction, query rewriting, code-based reasoning over retrieved context

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Spell       │  ← pyspellchecker fixes typos before embedding
│ Correction  │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Query       │  ← LLM rewrites follow-ups into standalone questions
│ Rewrite     │    (only for short, ambiguous follow-ups)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ FAISS       │  ← Embedding model converts query to vector,
│ Retrieval   │    FAISS returns top-k similar chunks
└─────┬───────┘
      │
      ▼
┌─────────────┐     ┌──────────────┐
│ Extract     │────▶│ Code-based   │  ← If dates found, Python evaluates
│ Facts (LLM) │     │ Evaluation   │    eligibility (no LLM math)
└─────┬───────┘     └──────┬───────┘
      │ (no dates)         │
      ▼                    ▼
┌──────────────────────────┐
│ Generate Answer (LLM)    │  ← Presents results naturally using
│                          │    retrieved context + code conclusions
└──────────────────────────┘
```

---

## Project Structure

```
.
├── notebook.ipynb          # Step-by-step RAG tutorial (start here)
├── app.py                  # Streamlit chatbot with LangGraph
├── data/
│   ├── Everstorm_Return.pdf
│   ├── Everstorm_Shipping.pdf
│   ├── Everstorm_Payment.pdf
│   └── Everstorm_Product.pdf
├── faiss_index/            # Generated — do not commit
│   ├── index.faiss
│   └── index.pkl
├── environment.yml         # Conda environment
├── .gitignore
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

### 2. Start Ollama and pull a model

Open a terminal and start the Ollama server (keep this running):

```bash
ollama serve
```

In a new terminal, pull the model:

```bash
ollama pull llama3.1:8b
```

> **Note:** The 8B model requires ~5GB of disk space and ~6GB of RAM. If your machine has limited resources, use `gemma3:4b` or `gemma3:1b` instead, and update the model name in `app.py` accordingly.

### 3. Build the FAISS index

Open and run `notebook.ipynb` from top to bottom. This will:

- Load the PDF documents from `data/`
- Chunk the text
- Generate embeddings using `gte-small`
- Build and save the FAISS index to `faiss_index/`

### 4. Run the chatbot

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How It Works

### Notebook (notebook.ipynb)

The notebook is the learning material. It builds the RAG pipeline incrementally:

**Data Preparation** — Uses `PyPDFLoader` to extract text from Everstorm Outfitters policy documents (synthetic data for learning). Falls back to local PDFs if web URLs are unavailable. Chunks documents using `RecursiveCharacterTextSplitter` with 300-character chunks and 40-character overlap.

**Vector Store** — Embeds chunks using HuggingFace's `gte-small` model (384 dimensions) and indexes them with FAISS. The index is saved locally as `index.faiss` (vectors) and `index.pkl` (document store + ID mapping).

**RAG Pipeline** — Two approaches are implemented side by side:
- `ManualRAG` — explicit function calls, manual chat history management
- `LangGraphRAG` — graph-based workflow with shared state, nodes, and edges

Both handle multi-turn conversations through query rewriting: follow-up questions like "how long do I have?" get rewritten as standalone queries using chat history context.

### Streamlit App (app.py)

The app wraps the LangGraph pipeline in a web interface with additional production-aware features:

**Spell Correction** — Uses `pyspellchecker` to fix typos before they hit the embedding model. "refund poolicy" becomes "refund policy" so FAISS retrieves the right chunks. Words containing numbers are skipped to preserve "40days", "2days", etc.

**Query Rewriting** — For short, ambiguous follow-ups (< 8 words), the LLM rewrites them as standalone questions. Longer, self-contained questions skip this step to avoid the LLM rewrite introducing errors.

**Cached Resources** — The embedding model, FAISS index, LLM connection, and graph are loaded once using `@st.cache_resource`. Streamlit reruns the entire script on every interaction, so caching prevents expensive reloads.

**Session State** — Chat history persists across Streamlit reruns using `st.session_state`. Two lists are maintained: one with LangChain message objects (for LangGraph), one with simple dicts (for Streamlit's chat UI).

---

## Key Lessons & Limitations

This project intentionally surfaces real-world RAG challenges:

**Small models struggle with reasoning.** Gemma 1B and even Llama 8B gave incorrect answers to date-comparison questions ("I ordered 40 days ago but received it yesterday — can I return?"). The model would correctly state "30 days from delivery" but then conclude "not eligible." The fix: extract facts with the LLM, evaluate logic in Python code, and have the LLM present the conclusion.

**Chunk size matters.** 300-character chunks retrieved the right content but sometimes returned sentence fragments. 800-character chunks provided better context but changed which chunks ranked highest. There's no universal right answer — test with your data.

**Spell correction has edge cases.** Dictionary-based correction can't handle domain-specific terms or severely mangled input. Words like "2days" (no space) need special handling to preserve numbers.

**Query rewriting is fragile with small models.** The 8B model sometimes answered the question instead of rewriting it, or added assumptions that biased retrieval. Strict prompts with length limits and GOOD/BAD examples help, but aren't foolproof.

**PDF extraction produces messy text.** Double spaces, broken lines, and lost formatting are common. Cleaning text with regex before embedding improves both retrieval quality and LLM comprehension.

---

## Customization

**Use a different model:**

```bash
ollama pull <model-name>
```

Update `load_llm()` in `app.py`:

```python
def load_llm():
    return ChatOllama(model="<model-name>", temperature=0.1)
```

**Use a different embedding model:**

Update both the notebook (when building the index) and `load_vectordb()` in `app.py`:

```python
emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

> **Important:** If you change the embedding model, you must rebuild the FAISS index by rerunning the notebook.

**Add your own documents:**

Place PDFs in `data/`, update the glob pattern in `load_offline_files()`, and rebuild the index.

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Embeddings | HuggingFace `gte-small` | Convert text to 384-dim vectors |
| Vector Store | FAISS | Fast similarity search |
| LLM | Llama 3.1 8B via Ollama | Local inference, no API keys |
| Orchestration | LangGraph | Stateful graph-based workflows |
| Framework | LangChain | Document loading, text splitting, retrieval |
| UI | Streamlit | Interactive chat interface |
| Spell Check | pyspellchecker | Pre-retrieval query correction |

---

## What's Next

If you want to take this further, here are production-grade improvements to explore:

- **Re-ranking** with cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve retrieval precision
- **LangGraph persistence** with SQLite/Postgres checkpointers for conversation history that survives server restarts
- **Evaluation** using RAGAS or LLM-as-judge to measure retrieval recall and answer quality
- **Streaming** responses token-by-token for better UX
- **Query routing** to skip retrieval for greetings and simple questions
- **API-based models** (GPT-4, Claude) for significantly better reasoning at the cost of API fees
