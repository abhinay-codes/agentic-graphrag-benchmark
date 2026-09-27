import requests
import json
import time
from typing import List, Union, Dict, Any, Optional
from src.llm.ollama_client import OllamaClient

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.default_model = model

    def embed(self, texts: Union[str, List[str]], model: str = "text-embedding-3-small") -> List[List[float]]:
        if isinstance(texts, str): texts = [texts]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        res = requests.post(f"{self.base_url}/embeddings", json={"input": texts, "model": model}, headers=headers, timeout=60)
        res.raise_for_status()
        return [item["embedding"] for item in res.json()["data"]]

    def generate(self, prompt: str, model: str = None, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, temperature: float = None) -> Dict[str, Any]:
        use_model = model if (model and model != "qwen3:8b") else self.default_model
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": use_model, "messages": messages}
        if temperature is not None: payload["temperature"] = temperature
        elif options and "temperature" in options: payload["temperature"] = options["temperature"]
        if options and "num_predict" in options: payload["max_tokens"] = options["num_predict"]

        t0 = time.time()
        res = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=300)
        res.raise_for_status()
        data = res.json()
        usage = data.get("usage", {})
        return {
            "response": data["choices"][0]["message"]["content"],
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
            "total_duration_ns": int((time.time() - t0) * 1e9)
        }

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.default_model = model

    def embed(self, texts: Union[str, List[str]], model: str = "text-embedding-004") -> List[List[float]]:
        if isinstance(texts, str): texts = [texts]
        embeddings = []
        for t in texts:
            res = requests.post(f"{self.base_url}/{model}:embedContent?key={self.api_key}", json={"model": f"models/{model}", "content": {"parts": [{"text": t}]}}, timeout=60)
            res.raise_for_status()
            embeddings.append(res.json()["embedding"]["values"])
        return embeddings

    def generate(self, prompt: str, model: str = None, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, temperature: float = None) -> Dict[str, Any]:
        use_model = model if (model and model != "qwen3:8b") else self.default_model
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if system: payload["systemInstruction"] = {"parts": [{"text": system}]}

        gen_config = {}
        if temperature is not None: gen_config["temperature"] = temperature
        elif options and "temperature" in options: gen_config["temperature"] = options["temperature"]
        if options and "num_predict" in options: gen_config["maxOutputTokens"] = options["num_predict"]
        if gen_config: payload["generationConfig"] = gen_config

        t0 = time.time()
        res = requests.post(f"{self.base_url}/{use_model}:generateContent?key={self.api_key}", json=payload, timeout=300)
        res.raise_for_status()
        data = res.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            text = ""

        usage = data.get("usageMetadata", {})
        return {
            "response": text,
            "prompt_eval_count": usage.get("promptTokenCount", 0),
            "eval_count": usage.get("candidatesTokenCount", 0),
            "total_duration_ns": int((time.time() - t0) * 1e9)
        }

class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = model

    def embed(self, texts: Union[str, List[str]], model: str = "claude-3-haiku-20240307") -> List[List[float]]:
        return OllamaClient().embed(texts)

    def generate(self, prompt: str, model: str = None, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, temperature: float = None) -> Dict[str, Any]:
        use_model = model if (model and model != "qwen3:8b") else self.default_model
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        messages = [{"role": "user", "content": prompt}]

        payload = {"model": use_model, "messages": messages, "max_tokens": 4096}
        if system: payload["system"] = system
        if temperature is not None: payload["temperature"] = temperature
        elif options and "temperature" in options: payload["temperature"] = options["temperature"]
        if options and "num_predict" in options: payload["max_tokens"] = options["num_predict"]

        t0 = time.time()
        res = requests.post(f"{self.base_url}/messages", json=payload, headers=headers, timeout=300)
        res.raise_for_status()
        data = res.json()

        usage = data.get("usage", {})
        return {
            "response": data["content"][0]["text"],
            "prompt_eval_count": usage.get("input_tokens", 0),
            "eval_count": usage.get("output_tokens", 0),
            "total_duration_ns": int((time.time() - t0) * 1e9)
        }

class OllamaClientProxy(OllamaClient):
    def __init__(self, base_url="http://localhost:11434", default_model="qwen3:8b"):
        super().__init__(base_url)
        self.default_model = default_model

    def generate(self, prompt: str, model: str = "qwen3:8b", system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, temperature: float = None) -> Dict[str, Any]:
        use_model = model if (model and model != "qwen3:8b") else self.default_model
        return super().generate(prompt, model=use_model, system=system, options=options, temperature=temperature)

def get_llm_client(provider: str = "ollama", base_url: str = None, api_key: str = None, model: str = None):
    provider = provider.lower() if provider else "ollama"
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o-mini")
    elif provider == "gemini":
        return GeminiClient(api_key=api_key, model=model or "gemini-1.5-flash")
    elif provider == "anthropic" or provider == "claude":
        return AnthropicClient(api_key=api_key, model=model or "claude-3-haiku-20240307")
    else:
        return OllamaClientProxy(base_url=base_url or "http://localhost:11434", default_model=model or "qwen3:8b")
