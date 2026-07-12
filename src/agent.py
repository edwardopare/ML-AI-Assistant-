from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from src.config import GEMINI_API_ENDPOINT, GEMINI_API_KEY, GEMINI_MODEL
from src.retrieval import Retriever

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


class GeminiClient:
    def __init__(self, model_name: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY, endpoint: str = GEMINI_API_ENDPOINT):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint

        if genai and api_key:
            genai.configure(api_key=api_key)

        if genai and endpoint:
            try:
                genai.configure(api_endpoint=endpoint)
            except Exception:
                pass

    def is_available(self) -> bool:
        return genai is not None

    def generate(self, prompt: str) -> str:
        if genai is None:
            raise RuntimeError("Gemini client is unavailable. Install google-generativeai or configure a compatible Gemini SDK.")

        response = genai.responses.create(model=self.model_name, input=prompt)
        return getattr(response, "output_text", None) or getattr(response, "response_text", None) or str(response)

    def rewrite_query(self, query: str) -> str:
        if not self.is_available():
            return query

        prompt = (
            "Rewrite the user question for retrieval over a local knowledge base. "
            "Keep the meaning and remove ambiguity.\n\n"
            f"Question: {query}\n\n"
            "Rewrite:" 
        )
        return self.generate(prompt).strip()

    def answer_question(self, question: str, documents: List[Document]) -> str:
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source')} page {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in documents
        )
        if self.is_available():
            prompt = (
                "You are an expert assistant with 45 years of experience. Use the retrieved document chunks to answer the question. "
                "Cite the source file name and page number for each fact you use.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:" 
            )
            return self.generate(prompt).strip()

        return (
            "[Gemini unavailable] Retrieved document chunks:\n\n"
            f"Question: {question}\n\n{context}"
        )


class RAGAgent:
    def __init__(self, top_k: int = 5) -> None:
        self.retriever = Retriever(top_k=top_k)
        self.gemini = GeminiClient()

    def answer(self, question: str) -> dict:
        rewritten = self.gemini.rewrite_query(question)
        documents = self.retriever.retrieve(rewritten)
        answer = self.gemini.answer_question(rewritten, documents)

        return {
            "query": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": [doc.metadata for doc in documents],
        }
