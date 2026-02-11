import streamlit as st
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ── Page Config ──
st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

# ── System Prompt ── --e


@st.cache_resource
def init_chain():
       vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small")
        allow_dangerous_desrialization=True,
       )

       retriever = vectordb.as_retriever(search_kwrags={"k": 8})
       llm = Ollama(model="gemma3:1b")


