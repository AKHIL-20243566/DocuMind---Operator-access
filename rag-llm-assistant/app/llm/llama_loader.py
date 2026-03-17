"""LLM loader — interface to Ollama or fallback."""
import logging
import requests

logger = logging.getLogger(__name__)


class LLMLoader:
    """Manages LLM connection and inference."""

    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="llama3", timeout=120):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        self._available = None

    def is_available(self):
        """Check if Ollama is running."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(
                self.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        logger.info(f"Ollama available: {self._available}")
        return self._available

    def generate(self, prompt):
        """Generate a response from the LLM."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.ConnectionError:
            logger.warning("Ollama not available")
            return None
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

    def generate_with_fallback(self, prompt, context=""):
        """Generate response, with a fallback if LLM is unavailable."""
        result = self.generate(prompt)
        if result is not None:
            return result
        if context:
            return f"Based on the retrieved documents:\n\n{context}\n\n*(Fallback mode — Ollama LLM is not connected.)*"
        return "I don't have enough information to answer that question. (LLM unavailable)"