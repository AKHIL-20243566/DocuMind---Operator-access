"""DocuMind RAG Library — LLM Loader (Ollama)
Owner: Aaron (Backend RAG Engineer)
Purpose: Standalone Ollama client used by the rag-llm-assistant modular library.
         Checks availability, generates responses (with and without context fallback).
Connection: Used by rag-llm-assistant/app/pipeline/rag_pipeline.py for LLM inference.
            Mirrors the logic in backend/llm.py but as a reusable class.
"""

import logging
import requests

logger = logging.getLogger(__name__)


class LLMLoader:
    """Manages LLM connection and inference via Ollama REST API.

    Owner: Aaron (Backend RAG Engineer)
    Provides:
    - is_available(): health check
    - generate(prompt): non-streaming inference
    - generate_with_fallback(prompt, context): safe generation with context fallback
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3", timeout: int = 120):
        self.base_url     = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.tags_url     = f"{base_url}/api/tags"
        self.model        = model
        self.timeout      = timeout

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and responding."""
        try:
            resp = requests.get(self.tags_url, timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                logger.info("Ollama available. Models: %s", [m["name"] for m in models])
                return True
            return False
        except Exception as e:
            logger.error("Ollama availability check failed: %s", e)
            return False

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the full response string.

        Raises RuntimeError if Ollama is unavailable.
        Raises on unexpected response format — errors are NOT silenced.
        """
        if not self.is_available():
            raise RuntimeError("Ollama is not running or unreachable")

        response = requests.post(
            self.generate_url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response format: {data}")

        return data["response"]

    def generate_with_fallback(self, prompt: str, context: str = "") -> str:
        """Attempt generation; on failure return retrieved context as fallback.

        Owner: Aaron — safe wrapper used when LLM downtime must not crash the pipeline.
        """
        try:
            return self.generate(prompt)
        except Exception as e:
            logger.warning("LLM unavailable (%s) — falling back to context display", e)
            if context:
                return (
                    "LLM unavailable. Showing retrieved context instead:\n\n"
                    f"{context}"
                )
            return "LLM is currently unavailable. Please check the Ollama server."
