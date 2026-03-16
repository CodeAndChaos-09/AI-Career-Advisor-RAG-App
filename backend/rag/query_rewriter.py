# backend/rag/query_rewriter.py

from transformers import pipeline

# Load lightweight model for rewriting
rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=64
)


def rewrite_query(question: str) -> str:
    """
    Rewrites the user query to improve retrieval quality.
    """

    prompt = f"""
Rewrite the following question to make it clearer and better for searching documents.

Question: {question}

Rewritten question:
"""

    try:
        result = rewriter(prompt)
        rewritten_query = result[0]["generated_text"]
        return rewritten_query.strip()

    except Exception:
        # fallback if rewriting fails
        return question