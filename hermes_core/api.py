"""LLM API client for experiment comparison.

Provider (Anthropic-compatible):
  - GLM-5.1 (智谱)

Also provides embedding via 智谱 API for vector similarity.
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

import anthropic


@dataclass
class LLMConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    use_auth_token: bool = False  # True = Bearer token, False = x-api-key


PROVIDERS = {
    "glm": LLMConfig(
        name="GLM-5.1",
        base_url="https://open.bigmodel.cn/api/anthropic",
        api_key="YOUR_API_KEY_HERE",
        model="GLM-5.1",
    ),
}


@dataclass
class CallStats:
    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "latency_ms": round(self.total_latency_ms, 0),
            "avg_latency_ms": round(self.total_latency_ms / max(self.calls, 1), 0),
        }


class LLMBenchmark:
    """Wraps an Anthropic-compatible LLM with call tracking."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.stats = CallStats()
        kwargs = {"base_url": config.base_url}
        if config.use_auth_token:
            kwargs["auth_token"] = config.api_key
        else:
            kwargs["api_key"] = config.api_key
        self._client = anthropic.Anthropic(**kwargs)

    def __call__(self, prompt: str, max_tokens: int = 500) -> str:
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = (time.perf_counter() - start) * 1000

            text = resp.content[0].text if resp.content else ""
            input_t = getattr(resp.usage, "input_tokens", 0) or 0
            output_t = getattr(resp.usage, "output_tokens", 0) or 0

            self.stats.calls += 1
            self.stats.total_input_tokens += input_t
            self.stats.total_output_tokens += output_t
            self.stats.total_latency_ms += latency

            return text
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.stats.calls += 1
            self.stats.total_latency_ms += latency
            return f"ERROR: {e}"


def get_embedding(text: str, model: str = "embedding-3") -> list[float]:
    """Get embedding vector from 智谱 API."""
    import http.client
    import hashlib

    # Use 智谱's native API for embeddings (not Anthropic-compatible)
    conn = http.client.HTTPSConnection("open.bigmodel.cn")
    payload = json.dumps({"model": model, "input": text})
    headers = {
        "Authorization": "Bearer YOUR_API_KEY_HERE",
        "Content-Type": "application/json",
    }
    conn.request("POST", "/api/paas/v4/embeddings", payload, headers)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    if "data" in data and len(data["data"]) > 0:
        return data["data"][0]["embedding"]
    raise ValueError(f"Embedding API error: {data}")


def get_all_providers() -> dict[str, LLMBenchmark]:
    """Return instantiated LLM benchmarks for all providers."""
    return {k: LLMBenchmark(v) for k, v in PROVIDERS.items()}
