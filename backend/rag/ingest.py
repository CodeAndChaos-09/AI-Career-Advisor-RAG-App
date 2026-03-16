# backend/rag/ingest.py

import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ----------------------------
# Paths
# ----------------------------

DATA_PATH = "./data"
VECTOR_DB_PATH = "vectorstore"


# ----------------------------
# Load Documents
# ----------------------------

def load_documents():

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    return documents


# ----------------------------
# Split Documents
# ----------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


# ----------------------------
# Create Embeddings
# ----------------------------

def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTOR_DB_PATH)

    print("Vector database created successfully!")


# ----------------------------
# Main Function
# ----------------------------

def main():

    documents = load_documents()

    chunks = split_documents(documents)

    create_vectorstore(chunks)


if __name__ == "__main__":
    main()