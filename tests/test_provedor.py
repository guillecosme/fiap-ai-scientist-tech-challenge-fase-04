"""Configuracao do provedor, cache e modelo de reserva, sem chamar a API."""

import pytest

from src.generation import provedor
from src.generation.provedor import LLM, model_name


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TC_ENV_FILE", raising=False)
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "modelo-principal")
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "modelo-juiz")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)


def test_model_names_come_from_environment():
    assert model_name("chat") == "modelo-principal"
    assert model_name("judge") == "modelo-juiz"


def test_missing_model_is_an_error(monkeypatch):
    monkeypatch.delenv("GEMINI_CHAT_MODEL")
    with pytest.raises(RuntimeError):
        model_name("chat")


def test_cache_avoids_a_second_call(monkeypatch, tmp_path):
    calls = []

    def fake_call(self, msgs, schema):
        calls.append(self.model)
        return "ok", {"input_tokens": 3, "output_tokens": 1}

    monkeypatch.setattr(provedor.LLM, "_call", fake_call)
    llm = LLM("chat", cache_dir=tmp_path / "cache", interval=0)
    assert llm.invoke("oi") == "ok" and llm.invoke("oi") == "ok"
    assert calls == ["modelo-principal"]
    assert llm.usage()["chamadas_novas"] == 1 and llm.usage()["chamadas"] == 2


def test_fallback_model_is_used_when_the_main_one_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "modelo-reserva")

    def fake_call(self, msgs, schema):
        if self.model == "modelo-principal":
            raise provedor.QuotaExhausted("cota")
        return "da reserva", {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(provedor.LLM, "_call", fake_call)
    llm = LLM("chat", cache_dir=tmp_path / "cache", interval=0)
    assert llm.invoke("oi") == "da reserva"
    assert llm.calls[-1].model == "modelo-reserva"
