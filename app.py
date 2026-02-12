import streamlit as st
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ── Config ──
MAX_HISTORY_TURNS = 10

# ── Page Config ──
st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

# ── System Prompt ──
SYSTEM_PROMPT = """You are a helpful and accurate Customer Support AI.
Your task is to answer the user's question using ONLY the provided CONTEXT.

### LOGIC GUIDELINES:
1. **The Anchor Date:** Check if the policy counts days from "Order Date" or "Delivery Date".
   - If the policy says "30 days from delivery":
   - And User says: "Ordered 40 days ago, received today."
   - Then: Time elapsed is 0 days. (Eligible).
   - DO NOT compare "Order Date" against the return window.

2. **Uncertainty:** If the CONTEXT does not contain the answer, simply say: "I'm sorry, but the provided documents do not contain that information."

3. **Tone:** Be professional, concise, and direct.

### CONTEXT:
{context}
"""

# ── Helper ──
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def trim_history(chat_history: list) -> list:
    """Keep only the last N turns (each turn = 1 Human + 1 AI Message)"""
    max_messages = MAX_HISTORY_TURNS * 2
    if len(chat_history) > max_messages:
        return chat_history[-max_messages:]
    return chat_history

# ── Load resources once ──
@st.cache_resource
def load_vectordb():
       try:
              emb_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
              vectordb = FAISS.load_local(
                     "faiss_index",
                     emb_model,
                     allow_dangerous_deserialization=True,
              )
              return vectordb

       except Exception as e:
           st.error(f"Failed to load the FAISS index: {e}")
           return None

@st.cache_resource
def load_llm():
    try:
       llm =  ChatOllama(model="llama3.1:8b", temperature=0)

       # Health check
       llm.invoke([HumanMessage(content="hello")])
       return llm

    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}")
        return None

@st.cache_resource
def build_graph():
    vectordb = load_vectordb()
    llm = load_llm()

    if vectordb is None or llm is None:
        return None

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # ── State ──
    class RAGState(TypedDict):
        question: str
        chat_history: list
        rewritten_query: str
        context: str
        source_documents: list
        answer: str

    # ── Nodes ──
    def rewrite_query_node(state: RAGState) -> dict:
        question = state["question"]

        # OPTIMIZATION: If no history, use raw question (prevents hallucination)
        if not state.get("chat_history", []):
            return {"rewritten_query": question}

        messages = [
            SystemMessage(content=(
                "You are a Search Query Optimizer. "
                "Rewrite the user's input into a specific, keyword-rich search query.\n"
                "RULES:\n"
                "1. Keep specific numbers (e.g. '40 days').\n"
                "2. Remove politeness ('Please tell me').\n"
                "3. Do NOT answer the question.\n"
                "4. Output ONLY the rewritten query, nothing else."
            )),
            *state["chat_history"],
            HumanMessage(content=f"User's new question: {question}"),
        ]

        try:
              response = llm.invoke(messages)
              clean_query = response.content.strip().replace('"', '')

              if (
                  len(clean_query) == 0
                  or len(clean_query) > 500
                  or "\n" in clean_query
              ):
                  print(f"[REWRITE FALLBACK] Bad rewrite: '{clean_query[:100]}' → using original")
                  return {"rewritten_query": question}    
              return {"rewritten_query": clean_query}
        except Exception as e:
            print(f"[REWRITE FALLBACK] Error: {e} → using original")
            return {"rewritten_query": question}

    def retrieve_node(state: RAGState) -> dict:
        docs = retriever.invoke(state["rewritten_query"])
        print("=========================================")
        return {
            "context": format_docs(docs),
            "source_documents": docs
        }

    def generate_node(state: RAGState) -> dict:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"])),
            *state["chat_history"],
            HumanMessage(content=state["question"])
        ]

        response = llm.invoke(messages)

        updated_history = state["chat_history"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]

        return {
            "answer": response.content,
            "chat_history": trim_history(updated_history),
            "source_documents": state["source_documents"] # Pass docs through
        }

    # ── Build Graph ──
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# ── Initialize ──
rag_chain = build_graph()


if rag_chain is None:
    st.warning("Chatbot is unavailable. Please ensure Ollama is running and FAISS index exists.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.display_messages = []

# ── Display chat history ──
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If this message has associated sources (saved previously), show them
        if "sources" in msg:
            with st.expander("View Source Documents"):
                for idx, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx+1}:**")
                    st.caption(doc.page_content[:300] + "...") 
                    st.divider()

if st.sidebar.button("New Conversation"):
    st.session_state.chat_history = []
    st.session_state.display_messages = []
    st.rerun()

# ── Handle new input ──
if question := st.chat_input("Ask a question"):
    st.chat_message("user").write(question)
    st.session_state.display_messages.append({"role": "user", "content": question})

    try:
       with st.spinner("Checking policies..."):
              result = rag_chain.invoke({
              "question": question,
              "chat_history": st.session_state.chat_history,
              "rewritten_query": "",
              "context": "",
              "source_documents": [],
              "answer": ""
              })

    except Exception as e:
        st.error(f"Something went wrong. {e}")
        st.stop()

    st.session_state.chat_history = result["chat_history"]

    # Show bot response
    with st.chat_message("assistant"):
        st.write(result["answer"])

        # ─── NEW: Show Sources ───
        if result["source_documents"]:
            with st.expander("View Source Documents"):
                for idx, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {idx+1}** (from {doc.metadata.get('source', 'unknown')})")
                    st.caption(doc.page_content) # Show full content or slice it
                    st.divider()

    # Save response AND sources to display history
    st.session_state.display_messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "sources": result["source_documents"]
    })
