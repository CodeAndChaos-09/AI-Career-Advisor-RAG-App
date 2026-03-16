# backend/rag/query_rewriter.py

def rewrite_query(question: str) -> str:
    """
    Simple query passthrough.
    We avoid loading heavy models here to keep deployment stable.
    """
    return question