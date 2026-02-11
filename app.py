import streamlit as st
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from spellchecker import SpellChecker

# ── Page Config ──
st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

# ── System Prompt ──
SYSTEM_PROMPT = """You are a Customer Support Chatbot.
Answer the user's question using the CONTEXT below.

When evaluating return eligibility:
- Identify the policy's return window and what date it counts from.
- Calculate: days since that date vs the allowed window.
- Think step by step before giving your conclusion.

Think step by step before answering. Be concise and cite sources when possible.

CONTEXT:
{context}"""

# ── Helper ──
def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)

def correct_spelling(text, spell):
       words = text.split()
       corrected = []

       for word in words:
               # Skip words with numbers — don't spellcheck "2days", "40", etc.
              if any(char.isdigit() for char in word):
                     corrected.append(word)
              else: 
                     corrected.append(spell.correction(word) or word)
       return " ".join(corrected)

# ── Load resources once ──
@st.cache_resource
def load_vectordb():
       emb_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
       vectordb = FAISS.load_local(
        "faiss_index",
        emb_model,
        allow_dangerous_deserialization=True,
       )

       return vectordb

@st.cache_resource
def load_spellchecker():
       return SpellChecker()

@st.cache_resource
def load_llm():
       return ChatOllama(model="llama3.1:8b", temperature=0.1)

@st.cache_resource
def build_graph():
       vectordb = load_vectordb()
       retriever = vectordb.as_retriever(search_kwargs={"k":12})
       llm = load_llm()
       spell = load_spellchecker()

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
              corrected = correct_spelling(state["question"], spell) 
              print(f"Original: {state['question']} → Corrected: {corrected}")

              if not state["chat_history"]:
                     return {"rewritten_query": corrected}

              messages = [
    SystemMessage(content=(
        "Your ONLY job is to rewrite the user's latest message as a "
        "standalone question using chat history for context.\n\n"
        "RULES:\n"
        "- Output must be a SINGLE question\n"
        "- Output must END with a question mark\n"
        "- Output must be UNDER 30 words\n"
        "- Do NOT answer the question\n"
        "- Do NOT include greetings or explanations\n\n"
        "GOOD: What is the contact information for customer support?\n"
        "BAD: Here is the contact information: phone 1-800..."
    )),
    *state["chat_history"],
    HumanMessage(content=corrected),
]

              rewritten_query = llm.invoke(messages)

              return {"rewritten_query": rewritten_query.content}

       def retrieve_node(state: RAGState) -> dict:
              docs = retriever.invoke(state["rewritten_query"])
              print(f"Query: {state['rewritten_query']}")
              print(f"Top chunk: {docs[0].page_content[:200]}")

              return {
                     "context": format_docs(docs),
                     "source_documents": docs
              }

       def generate_nodes(state: RAGState) -> dict:
              messages = [
                     SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"])),
                     *state["chat_history"],
                     HumanMessage(content=state["question"])
              ]

              print(f"\n=== FULL CONTEXT ===\n{state['context'][:1000]}")
              print(f"\n=== QUESTION ===\n{state['question']}")

              response = llm.invoke(messages)

              updated_history = state["chat_history"] + [
                     HumanMessage(content=state["question"]),
                     AIMessage(content=response.content)
              ]

              return {
                     "answer": response.content,
                     "chat_history": updated_history
              }

       # ── Build Graph ──
       graph = StateGraph(RAGState)
       graph.add_node("rewrite", rewrite_query_node)
       graph.add_node("retrieve", retrieve_node)
       graph.add_node("generate", generate_nodes)

       graph.add_edge(START, "rewrite")
       graph.add_edge("rewrite", "retrieve")
       graph.add_edge("retrieve", "generate")
       graph.add_edge("generate", END)

       return graph.compile()

# ── Initialize ──
rag_chain = build_graph()

if "chat_history" not in st.session_state:
       st.session_state.chat_history = []
       st.session_state.display_messages = []

# ── Display chat history ──
for msg in st.session_state.display_messages:
       st.chat_message(msg["role"]).write(msg["content"])

if st.sidebar.button("New Conversation"):
    st.session_state.chat_history = []
    st.session_state.display_messages = []
    st.rerun()


# ── Handle new input ──
if question := st.chat_input("Ask a question"):
       # Show user messahe
       st.chat_message("user").write(question)
       st.session_state.display_messages.append({"role": "user", "content": question})

       with st.spinner("Thinking..."):
              result = rag_chain.invoke({
                     "question": question,
                     "chat_history": st.session_state.chat_history,
                     "rewritten_query": "",
                     "context": "",
                     "source_documents": [],
                     "answer": ""
              })

       #|print(result["answer"])

       st.session_state.chat_history = result["chat_history"]

       # Show bot response
       st.chat_message("assistant").write(result["answer"])
       st.session_state.display_messages.append({"role":"assistant", "content": result["answer"]})
