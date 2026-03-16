# backend/rag/query_engine.py

import os

from rag.query_rewriter import rewrite_query
from rag.reranker import rerank_documents

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline


# -----------------------------
# Resolve correct project paths
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
# Load Vector Database
# -----------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# Load Local Language Model
# -----------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)


# -----------------------------
# Main RAG Function
# -----------------------------

def query_rag(question: str):

    # Step 1: Rewrite the query
    rewritten_query = rewrite_query(question)

    # Step 2: Retrieve documents
    docs = retriever.get_relevant_documents(rewritten_query)

    # Step 3: Rerank documents
    reranked_docs = rerank_documents(rewritten_query, docs)

    # Step 4: Prepare context
    context = "\n".join([doc.page_content for doc in reranked_docs])

    # Step 5: Build prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{rewritten_query}

Answer:
"""

    # Step 6: Generate response
    result = generator(prompt)

    answer = result[0]["generated_text"]

    return answer