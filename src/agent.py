from __future__ import annotations
import os
import requests
import json
import time
from typing import List, Generator
from pathlib import Path
from langchain_core.documents import Document
from .retrieval import Retriever


class OllamaClient:
    """
    Client for interacting with local Ollama API.
    Handles model inference, auto-detection, and streaming responses.
    """

    def __init__(self, model_name: str | None = None, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        self.client_available = self._check_availability()
        self.model_name = model_name or self._auto_detect_model()

    def _check_availability(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _auto_detect_model(self) -> str:
        """Auto-detect and select the fastest available Ollama model based on priority list."""
        if not self.client_available:
            return "mistral"  # fallback
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    # Prefer smaller/faster models: phi3 > orca-mini > llama3 > mistral
                    model_names = [m["name"].split(":")[0] for m in models]
                    for preferred in ["phi3", "orca-mini", "neural-chat", "llama3", "mistral"]:
                        if preferred in model_names:
                            print(f"[Ollama] Auto-detected model: {preferred}")
                            return preferred
                    # Fallback to first available
                    return model_names[0]
        except Exception:
            pass
        
        return "mistral"

    def is_available(self) -> bool:
        """Check if Ollama client is available."""
        return self.client_available

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        """Generate response from Ollama, optionally streaming chunks."""
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
                    "stream": stream,
                },
                timeout=300,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    return self._stream_response(response)
                else:
                    return response.json().get("response", "").strip()
            else:
                print(f"[Ollama Error] Status {response.status_code}: {response.text}", flush=True)
                return ""
        except Exception as e:
            print(f"[Ollama Error] {type(e).__name__}: {str(e)}", flush=True)
            return ""

    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response line by line from Ollama."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    continue

    def answer_question(self, question: str, documents: List[Document], stream: bool = False) -> str | Generator[str, None, None]:
        """Generate answer from retrieved documents, optionally streaming."""
        # Optimize context: extract top sentences instead of full chunks
        context_parts = []
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            
            # Limit to first 300 chars per document for efficiency
            truncated = content[:300] + ("..." if len(content) > 300 else "")
            context_parts.append(f"Source: {source} page {page}\n{truncated}")
        
        context = "\n\n".join(context_parts)
        
        if self.is_available():
            prompt = (
                "You are a helpful assistant. Answer the question using ONLY the provided context. "
                "Always cite the source file and page number. Be concise.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:" 
            )
            
            if stream:
                return self.generate(prompt, stream=True)
            else:
                answer = self.generate(prompt, stream=False)
                if not answer:
                    return f"[No answer generated] Retrieved context:\n{context}"
                return answer

        return (
            "[Ollama unavailable] Retrieved document chunks:\n\n"
            f"Question: {question}\n\n{context}"
        )


class ResponseCache:
    """
    Simple JSON-based response cache for storing and retrieving query results.
    Enables instant responses for repeated queries.
    """

    def __init__(self, cache_dir: Path = Path("data/.cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "query_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persist cache to JSON file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[Cache] Failed to save: {e}")

    def get(self, query: str) -> dict | None:
        """Get cached response for query."""
        key = query.lower().strip()
        return self.cache.get(key)

    def set(self, query: str, result: dict) -> None:
        """Cache a response."""
        key = query.lower().strip()
        self.cache[key] = result
        self._save_cache()

    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        self._save_cache()


class RAGAgent:
    """
    Main RAG agent that orchestrates retrieval and generation.
    Handles document retrieval, LLM inference, caching, and performance tracking.
    """

    def __init__(self, top_k: int = 2, model_name: str | None = None, use_cache: bool = True, stream: bool = False) -> None:
        self.retriever = Retriever(top_k=top_k)
        self.llm = OllamaClient(model_name=model_name)
        self.cache = ResponseCache() if use_cache else None
        self.stream = stream

    def answer(self, question: str) -> dict:
        """
        Answer a question using retrieval and generation.
        Returns cached result if available, otherwise runs full pipeline.
        """
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                print(f"[Cache Hit] Retrieved from cache", flush=True)
                return cached
        
        # Retrieve documents (skip query rewriting for speed)
        retrieval_start = time.time()
        documents = self.retriever.retrieve(question)
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        
        if self.stream:
            # Stream response
            answer = ""
            print("\n[Streaming Answer]", flush=True)
            for chunk in self.llm.answer_question(question, documents, stream=True):
                print(chunk, end="", flush=True)
                answer += chunk
            print("\n", flush=True)
        else:
            answer = self.llm.answer_question(question, documents, stream=False)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        result = {
            "query": question,
            "answer": answer,
            "sources": [doc.metadata for doc in documents],
            "metrics": {
                "retrieval_time_s": round(retrieval_time, 2),
                "generation_time_s": round(generation_time, 2),
                "total_time_s": round(total_time, 2),
            }
        }
        
        # Cache the result
        if self.cache:
            self.cache.set(question, result)
        
        return result

