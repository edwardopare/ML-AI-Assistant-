from __future__ import annotations
import os
import requests
from typing import List
from langchain_core.documents import Document
from .retrieval import Retriever


class OllamaClient:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        self.client_available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        return self.client_available

    def generate(self, prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                f"Ollama is unavailable at {self.base_url}. Make sure Ollama is running."
            )

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"[Ollama Error] Status {response.status_code}: {response.text}", flush=True)
                return ""
        except Exception as e:
            # On any client/runtime error, print debug info and return empty string
            print(f"[Ollama Error] {type(e).__name__}: {str(e)}", flush=True)
            return ""

    def rewrite_query(self, query: str) -> str:
        if not self.is_available():
            return query

        prompt = (
            "Rewrite the user question for retrieval over a local knowledge base. "
            "Keep the meaning and remove ambiguity.\n\n"
            f"Question: {query}\n\n"
            "Rewrite:" 
        )
        rewritten = self.generate(prompt).strip()
        # If generation fails, return original query
        return rewritten if rewritten else query

    def answer_question(self, question: str, documents: List[Document]) -> str:
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source')} page {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in documents
        )
        if self.is_available():
            prompt = (
                "You are an expert assistant. Use the retrieved document chunks to answer the question. "
                "Cite the source file name and page number for each fact you use.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:" 
            )
            answer = self.generate(prompt).strip()
            if not answer:
                return f"[No answer generated] Retrieved context:\n{context}"
            return answer

        return (
            "[Ollama unavailable] Retrieved document chunks:\n\n"
            f"Question: {question}\n\n{context}"
        )


class RAGAgent:
    def __init__(self, top_k: int = 2, model_name: str = "mistral") -> None:
        self.retriever = Retriever(top_k=top_k)
        self.llm = OllamaClient(model_name=model_name)

    def answer(self, question: str) -> dict:
        rewritten = self.llm.rewrite_query(question)
        documents = self.retriever.retrieve(rewritten)
        answer = self.llm.answer_question(rewritten, documents)

        return {
            "query": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": [doc.metadata for doc in documents],
        }
