# backend/rag/query_engine.py

import os

from rag.query_rewriter import rewrite_query
from rag.reranker import rerank_documents

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline


# -----------------------------
# Resolve Paths
# -----------------------------

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
VECTORSTORE_PATH = os.path.join(BACKEND_DIR, "vectorstore")


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load Vector Store
# -----------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# Load LLM
# -----------------------------

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)


# -----------------------------
# Main RAG Pipeline
# -----------------------------

def query_rag(question: str):

    # Step 1: Rewrite Query
    rewritten_query = rewrite_query(question)

    # Step 2: Retrieve Documents
    docs = retriever.get_relevant_documents(rewritten_query)

    # Step 3: Rerank Documents
    reranked_docs = rerank_documents(rewritten_query, docs)

    context = "\n".join([doc.page_content for doc in reranked_docs])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{rewritten_query}

Answer:
"""

    result = generator(prompt)

    answer = result[0]["generated_text"]

    return answer