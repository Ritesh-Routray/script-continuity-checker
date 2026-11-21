"""Model-agnostic JSON chat. Swap providers with one env var / CLI flag.

    LLM_PROVIDER = gemini | groq | openai   (openai = any OpenAI-compatible endpoint:
                                             OpenRouter, Ollama, vLLM, a Kuku gateway...)
    LLM_MODEL    = optional model override
Keys: GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY (+ OPENAI_BASE_URL).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
}
CACHE_DIR = Path(os.getenv("CC_CACHE_DIR", Path(__file__).resolve().parent.parent / ".llm_cache"))


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float = 0.0, use_cache: bool = True, min_interval: float | None = None):
        load_dotenv()
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError(f"Unknown provider '{self.provider}'. Use one of {list(DEFAULT_MODELS)}")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.temperature = temperature
        self.use_cache = use_cache
        # free tiers are rate limited; space calls out (seconds between requests)
        self.min_interval = min_interval if min_interval is not None else \
            {"gemini": 6.5, "groq": 2.5}.get(self.provider, 0.0)
        self._last = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        key_var = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}[self.provider]
        self.key = os.getenv(key_var) or (os.getenv("GOOGLE_API_KEY") if self.provider == "gemini" else None)
        if not self.key and not (self.provider == "openai" and os.getenv("OPENAI_BASE_URL")):
            raise LLMError(f"{key_var} is not set (put it in continuity_checker/.env)")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------ public
    def json(self, system: str, user: str, max_tokens: int = 8192) -> dict:
        key = hashlib.sha256(f"{self.name}|{self.temperature}|{system}|{user}".encode()).hexdigest()[:24]
        cache_file = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))["parsed"]
        raw = self._call_with_retry(system, user, max_tokens)
        try:
            parsed = parse_json(raw)
        except LLMError:   # truncated / malformed output: ask once more, more tersely
            raw = self._call_with_retry(system, user + "\n\nReturn compact, complete, valid JSON. "
                                        "Keep summaries short.", max_tokens)
            parsed = parse_json(raw)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"model": self.name, "raw": raw, "parsed": parsed},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
        return parsed

    # ------------------------------------------------------------------ internals
    def _call_with_retry(self, system: str, user: str, max_tokens: int) -> str:
        delay = 10.0
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                out = (self._gemini if self.provider == "gemini" else self._openai_compat)(system, user, max_tokens)
                self.calls += 1
                return out
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code in (429, 500, 502, 503, 504) and attempt < 5:
                    retry_after = _retry_after(e.response)
                    time.sleep(retry_after or delay)
                    delay = min(delay * 2, 90)
                    continue
                body = e.response.text[:500] if e.response is not None else ""
                raise LLMError(f"{self.name} HTTP {code}: {body}") from e
            except requests.RequestException as e:
                if attempt < 5:
                    time.sleep(delay)
                    continue
                raise LLMError(str(e)) from e
        raise LLMError("retries exhausted")

    def _gemini(self, system: str, user: str, max_tokens: int) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }
        if "2.5" in self.model:   # don't spend the output budget on hidden reasoning
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0 if "flash" in self.model else 512}
        r = requests.post(url, json=body, headers={"x-goog-api-key": self.key}, timeout=180)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usageMetadata", {})
        self.tokens_in += usage.get("promptTokenCount", 0)
        self.tokens_out += usage.get("candidatesTokenCount", 0)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"Gemini returned no text: {json.dumps(data)[:400]}")

    def _openai_compat(self, system: str, user: str, max_tokens: int) -> str:
        base = "https://api.groq.com/openai/v1" if self.provider == "groq" else \
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        if self.provider == "groq":
            # Groq reserves max_tokens against the per-minute token budget (8k on free tier)
            max_tokens = min(max_tokens, int(os.getenv("GROQ_MAX_TOKENS", "3000")))
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = requests.post(base.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=180)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


def _retry_after(resp) -> float | None:
    if resp is None:
        return None
    h = resp.headers.get("retry-after")
    if h:
        try:
            return float(h) + 1
        except ValueError:
            pass
    m = re.search(r"retry in ([\d.]+)s|try again in (?:(\d+)m)?([\d.]+)s", resp.text)
    if m:
        if m.group(1):
            return float(m.group(1)) + 1
        return int(m.group(2) or 0) * 60 + float(m.group(3)) + 1
    return None


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON: {text[:300]}")
