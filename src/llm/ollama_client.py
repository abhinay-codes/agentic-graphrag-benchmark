import requests
from typing import List, Union, Dict, Any, Optional
import time

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.last_error = None

    def check_health(self) -> bool:
        """Checks if the Ollama server is running and accessible."""
        self.clear_error()
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            res.raise_for_status()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def get_models(self) -> List[str]:
        """Returns a list of installed model names."""
        self.clear_error()
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            res.raise_for_status()
            return [model["name"] for model in res.json().get("models", [])]
        except Exception as e:
            self.last_error = str(e)
            return []

    def embed(self, texts: Union[str, List[str]], model: str = "nomic-embed-text") -> List[List[float]]:
        """
        Embeds a single string or a batch of strings using /api/embed.
        Ollama natively supports batching if 'input' is a list of strings.
        """
        self.clear_error()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        payload = {
            "model": model,
            "input": texts
        }

        try:
            res = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            embeddings = data.get("embeddings", [])
            return embeddings
        except Exception as e:
            self.last_error = str(e)
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def generate(self, prompt: str, model: str = "qwen3:8b", system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, temperature: float = None) -> Dict[str, Any]:
        """Generates text from a prompt using /api/generate.
        Returns a dictionary with 'response' and token usage metrics.
        """
        self.clear_error()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            if options is None:
                options = {}
            options.setdefault("temperature", temperature)
        if options:
            payload["options"] = options

        try:
            res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
            res.raise_for_status()
            data = res.json()
            return {
                "response": data.get("response", ""),
                "prompt_eval_count": data.get("prompt_eval_count", None),
                "eval_count": data.get("eval_count", None),
                "total_duration_ns": data.get("total_duration", None)
            }
        except Exception as e:
            self.last_error = str(e)
            raise RuntimeError(f"Ollama generation failed: {e}")

    def clear_error(self):
        """Clears the last recorded error."""
        self.last_error = None
