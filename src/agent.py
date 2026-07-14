from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from .config import GEMINI_API_ENDPOINT, GEMINI_API_KEY, GEMINI_MODEL
from .retrieval import Retriever

try:
    import google.genai as genai
    from google.genai.types import HttpOptions
except ImportError:  # pragma: no cover
    genai = None
    HttpOptions = None


class GeminiClient:
    def __init__(self, model_name: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY, endpoint: str = GEMINI_API_ENDPOINT):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint

        if genai is None:
            self.client = None
            return

        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if endpoint and HttpOptions is not None:
            client_kwargs['http_options'] = HttpOptions(base_url=endpoint)

        # Only instantiate the client when there's an explicit credential or endpoint configured.
        if client_kwargs:
            try:
                self.client = genai.Client(**client_kwargs)
            except Exception:
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def generate(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "Gemini client is unavailable. Install google-genai or configure a compatible Gemini SDK."
            )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text or str(response)
        except Exception:
            # On any client/runtime error, fall back to returning the prompt
            # which causes the agent to use the original query or local context.
            return prompt

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
    def __init__(self, top_k: int = 2) -> None:
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
