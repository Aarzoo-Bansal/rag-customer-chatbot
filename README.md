# Customer Support RAG Chatbot

A hands-on project for learning how to build Retrieval-Augmented Generation (RAG) pipelines from scratch. It starts with a basic RAG implementation in a Jupyter notebook and evolves into a production-aware Streamlit chatbot with query rewriting and conversational memory.

Built with LangChain, LangGraph, FAISS, and Ollama — everything runs locally, no API keys needed.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.11-red)

## Demo

![Customer Support Chatbot Demo](https://github.com/Aarzoo-Bansal/rag-customer-chatbot/blob/main/ChatBot_GIF.gif)

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
10. **Production patterns** — error handling, query rewriting, sliding window history

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Query       │  ← LLM rewrites follow-ups into standalone questions
│ Rewrite     │    (skipped if no chat history exists)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ FAISS       │  ← Embedding model converts query to vector,
│ Retrieval   │    FAISS returns top-5 similar chunks
└─────┬───────┘
      │
      ▼
┌──────────────────────────┐
│ Generate Answer (LLM)    │  ← Produces response using retrieved
│                          │    context + conversation history
└──────────────────────────┘
```

---

## Project Structure

```
.
├── rag_chatbot.ipynb          # Step-by-step RAG tutorial (start here)
├── app.py                  # Streamlit chatbot with LangGraph
├── data/
│   ├── Everstorm_Return.pdf
│   ├── Everstorm_Shipping.pdf
│   ├── Everstorm_Payment.pdf
│   └── Everstorm_Product.pdf
├── faiss_index/            # This will be generated
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

The app wraps the LangGraph pipeline in a web interface with production-aware patterns:

**Error Handling** — Startup checks verify that both the FAISS index and Ollama connection are available. If either is missing, the app shows a user-friendly warning instead of crashing. Runtime errors during generation are also caught and displayed gracefully.

**Query Rewriting** — When chat history exists, the LLM rewrites follow-up questions into standalone search queries before retrieval. If no history exists, the raw question is used directly to avoid unnecessary LLM calls. A fallback mechanism catches bad rewrites (empty, too long, or multi-line) and reverts to the original question.

**Sliding Window History** — Chat history is capped at the last 10 turns (20 messages) to keep inference fast and prevent context window overflow. The full conversation remains visible in the UI — only the LLM-facing history is trimmed.

**Cached Resources** — The embedding model, FAISS index, LLM connection, and graph are loaded once using `@st.cache_resource`. Streamlit reruns the entire script on every interaction, so caching prevents expensive reloads.

**Session State** — Chat history persists across Streamlit reruns using `st.session_state`. Two lists are maintained: one with LangChain message objects (for LangGraph), one with simple dicts (for Streamlit's chat UI).

---

## Key Lessons & Limitations

This project intentionally surfaces real-world RAG challenges:

**Small models struggle with reasoning.** Gemma 1B and even Llama 8B gave incorrect answers to date-comparison questions ("I ordered 40 days ago but received it yesterday — can I return?"). The model would correctly state "30 days from delivery" but then conclude "not eligible." Careful system prompts with explicit logic guidelines help, but don't fully solve this.

**Chunk size matters.** 300-character chunks retrieved the right content but sometimes returned sentence fragments. 800-character chunks provided better context but changed which chunks ranked highest. There's no universal right answer — test with your data.

**Query rewriting is fragile with small models.** The 8B model sometimes answered the question instead of rewriting it, or added assumptions that biased retrieval. Strict prompts with explicit rules help, but a fallback to the original query is essential.

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

---