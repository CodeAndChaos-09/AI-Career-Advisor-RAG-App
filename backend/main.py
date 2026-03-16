from fastapi import FastAPI
from pydantic import BaseModel

from rag.query_engine import query_rag

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/ask")
def ask_question(data: Question):

    answer = query_rag(data.question)

    return {"answer": answer}