"""Acesso ao LLM: Gemini via LangChain, com cache em disco, retentativa e ritmo.

Configuracao por variaveis de ambiente (ver .env.example):
- GOOGLE_API_KEY: chave da API do Gemini;
- GEMINI_CHAT_MODEL: modelo que gera respostas, extrai filtros e escreve o HyDE;
- GEMINI_JUDGE_MODEL: modelo que julga relevancia e fidelidade (diferente do gerador);
- TC_LLM_INTERVALO: segundos minimos entre chamadas ao mesmo modelo (padrao 4).

As variaveis vem do ambiente, do arquivo apontado por TC_ENV_FILE ou do .env na raiz do
repositorio, nessa ordem. A chave nunca e registrada em log nem no cache.

Cada chamada e guardada em .cache/llm/ pelo hash de modelo, parametros, schema e
mensagens; repetir a mesma chamada nao gasta cota. Erros 429 e 5xx sao repetidos com
espera crescente.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

CACHE_DIR = Path(".cache/llm")
DEFAULT_INTERVAL = 4.0
RETRY_WAITS = [10, 20, 40, 80, 160]


def load_env() -> None:
    """Carrega as variaveis do arquivo indicado em TC_ENV_FILE e do .env local."""
    extra = os.environ.get("TC_ENV_FILE")
    if extra and Path(extra).expanduser().exists():
        load_dotenv(Path(extra).expanduser(), override=False)
    load_dotenv(Path(".env"), override=False)


def model_name(role: str) -> str:
    load_env()
    var = {"chat": "GEMINI_CHAT_MODEL", "judge": "GEMINI_JUDGE_MODEL"}[role]
    name = os.environ.get(var)
    if not name:
        raise RuntimeError(f"defina {var} no ambiente ou no .env")
    return name


class QuotaExhausted(RuntimeError):
    """Cota diaria do modelo esgotada: nao adianta repetir agora."""


@dataclass
class CallRecord:
    model: str
    cached: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    key: str


@dataclass
class LLM:
    role: str = "chat"
    model_override: str | None = None
    temperature: float = 0.0
    thinking_level: str | None = "low"
    timeout: float = 90.0
    cache_dir: Path = CACHE_DIR
    interval: float = field(default_factory=lambda: float(
        os.environ.get("TC_LLM_INTERVALO", DEFAULT_INTERVAL)))
    calls: list[CallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        load_env()
        self.model = self.model_override or model_name(self.role)
        self._client = None
        self._last_call = 0.0

    def _chat(self):
        if self._client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = {"model": self.model, "temperature": self.temperature,
                      "max_retries": 0, "timeout": self.timeout}
            if self.thinking_level and self.model.startswith("gemini-3"):
                kwargs["thinking_level"] = self.thinking_level
            self._client = ChatGoogleGenerativeAI(**kwargs)
        return self._client

    @staticmethod
    def _messages(prompt: str | list[BaseMessage], system: str | None) -> list[BaseMessage]:
        msgs = [HumanMessage(prompt)] if isinstance(prompt, str) else list(prompt)
        return ([SystemMessage(system)] if system else []) + msgs

    def _key(self, msgs: list[BaseMessage], schema: type[BaseModel] | None) -> str:
        payload = {
            "model": self.model, "temperature": self.temperature,
            "thinking": self.thinking_level,
            "schema": schema.model_json_schema() if schema else None,
            "messages": [(m.type, m.content) for m in msgs],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                              .encode()).hexdigest()

    def _wait_turn(self) -> None:
        wait = self.interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, msgs: list[BaseMessage], schema: type[BaseModel] | None) -> tuple[dict, dict]:
        runnable = (self._chat().with_structured_output(schema, include_raw=True)
                    if schema else self._chat())
        for attempt, wait in enumerate([0, *RETRY_WAITS]):
            if wait:
                time.sleep(wait)
            self._wait_turn()
            try:
                out = runnable.invoke(msgs)
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                if "PerDay" in text or "per_day" in text.lower():
                    raise QuotaExhausted(f"{self.model}: cota diaria esgotada") from None
                transient = any(code in text for code in
                                ("429", "500", "502", "503", "504", "DEADLINE", "UNAVAILABLE",
                                 "timed out", "Timeout"))
                if transient and attempt < len(RETRY_WAITS):
                    continue
                raise
            if schema:
                if out["parsing_error"] is not None or out["parsed"] is None:
                    if attempt < len(RETRY_WAITS):
                        continue
                    raise ValueError(f"saida fora do schema {schema.__name__}")
                raw, result = out["raw"], out["parsed"].model_dump()
            else:
                raw = out
                content = out.content
                result = content if isinstance(content, str) else "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            return result, dict(raw.usage_metadata or {})
        raise RuntimeError("tentativas esgotadas")

    def invoke(self, prompt: str | list[BaseMessage], schema: type[BaseModel] | None = None,
               system: str | None = None):
        """Texto (sem schema) ou instancia do schema; usa o cache quando existe."""
        msgs = self._messages(prompt, system)
        key = self._key(msgs, schema)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.calls.append(CallRecord(self.model, True, 0.0, data["usage"].get(
                "input_tokens", 0), data["usage"].get("output_tokens", 0), key))
        else:
            t0 = time.perf_counter()
            result, usage = self._call(msgs, schema)
            latency = time.perf_counter() - t0
            data = {"model": self.model, "result": result, "latency_s": latency,
                    "usage": {k: usage.get(k, 0) for k in ("input_tokens", "output_tokens")}}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            self.calls.append(CallRecord(self.model, False, latency, data["usage"][
                "input_tokens"], data["usage"]["output_tokens"], key))
        return schema.model_validate(data["result"]) if schema else data["result"]

    def usage(self) -> dict:
        new = [c for c in self.calls if not c.cached]
        return {"chamadas": len(self.calls), "chamadas_novas": len(new),
                "tokens_entrada": sum(c.input_tokens for c in self.calls),
                "tokens_saida": sum(c.output_tokens for c in self.calls),
                "latencia_media_s": (sum(c.latency_s for c in new) / len(new)) if new else 0.0}
