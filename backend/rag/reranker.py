# backend/rag/reranker.py

from sentence_transformers import SentenceTransformer, util

# Load reranker model
reranker_model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank_documents(query, docs, top_k=3):
    """
    Reranks retrieved documents based on semantic similarity
    """

    if not docs:
        return []

    query_embedding = reranker_model.encode(query, convert_to_tensor=True)

    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = reranker_model.encode(doc_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored_docs[:top_k]]

    return top_docs