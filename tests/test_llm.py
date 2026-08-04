"""Tests for llm.embed backend selection (vLLM / Azure)."""

from __future__ import annotations

import pytest

from rageval import llm


# ---------- backend resolution ----------

def test_resolve_embedding_backend_default_vllm(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert llm._resolve_embedding_backend(None, None) == "vllm"


def test_resolve_embedding_backend_env_azure(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    assert llm._resolve_embedding_backend(None, None) == "azure"


def test_resolve_embedding_backend_env_case_insensitive(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "  Azure ")
    assert llm._resolve_embedding_backend(None, None) == "azure"


def test_resolve_embedding_backend_model_prefix(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert llm._resolve_embedding_backend("azure/text-embedding-3-small", None) == "azure"


def test_resolve_embedding_backend_explicit_wins(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    assert llm._resolve_embedding_backend(None, "vllm") == "vllm"


# ---------- vLLM モデル別エンドポイント振り分け ----------

def test_vllm_endpoint_default(monkeypatch):
    monkeypatch.delenv("VLLM_ENDPOINT", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_MODEL_ENDPOINTS", raising=False)
    assert llm._vllm_endpoint() == "http://localhost:8000/v1"
    assert llm._vllm_endpoint("google/gemma-4-26B-A4B-it") == "http://localhost:8000/v1"


def test_vllm_endpoint_model_map(monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv(
        "VLLM_MODEL_ENDPOINTS",
        '{"nvidia/nemotron-judge": "http://localhost:8001/v1"}',
    )
    # 振り分け表にあるモデルは判定サーバへ、無いモデルは既定へ
    assert llm._vllm_endpoint("nvidia/nemotron-judge") == "http://localhost:8001/v1"
    assert llm._vllm_endpoint("google/gemma-4-26B-A4B-it") == "http://localhost:8000/v1"
    assert llm._vllm_endpoint() == "http://localhost:8000/v1"


def test_vllm_endpoint_map_invalid_json_raises(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL_ENDPOINTS", "{broken")
    with pytest.raises(llm.LLMError):
        llm._vllm_endpoint("any-model")


def test_vllm_endpoint_map_must_be_object(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL_ENDPOINTS", '["not", "a", "dict"]')
    with pytest.raises(llm.LLMError):
        llm._vllm_endpoint("any-model")


def test_generate_routes_model_to_mapped_endpoint(monkeypatch):
    captured = {}

    def fake_call(prompt, model, base_url, api_key, temperature, max_tokens,
                  json_mode, system, timeout, allow_thinking=False):
        captured["model"] = model
        captured["base_url"] = base_url
        return "ok"

    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv(
        "VLLM_MODEL_ENDPOINTS",
        '{"nvidia/nemotron-judge": "http://localhost:8001/v1"}',
    )
    monkeypatch.setattr(llm, "_call_openai_compatible", fake_call)

    out = llm.generate("hello", model="nvidia/nemotron-judge")
    assert out == "ok"
    assert captured["base_url"] == "http://localhost:8001/v1"
    assert captured["model"] == "nvidia/nemotron-judge"


# ---------- 判定モデルの既定値 (JUDGE_MODEL) ----------

def test_judge_default_prefers_judge_model_env(monkeypatch):
    from rageval.cli import _judge_default
    monkeypatch.setenv("JUDGE_MODEL", "nvidia/nemotron-judge")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    assert _judge_default("gpt-4.1-mini") == "nvidia/nemotron-judge"


def test_judge_default_falls_back_to_azure_then_literal(monkeypatch):
    from rageval.cli import _judge_default
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "my-azure-deploy")
    assert _judge_default("gpt-4.1-mini") == "my-azure-deploy"
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
    assert _judge_default("gpt-4.1-mini") == "gpt-4.1-mini"


# ---------- dispatch ----------

def test_embed_dispatches_to_azure(monkeypatch):
    captured = {}

    def fake_azure(texts, deployment, timeout):
        captured["deployment"] = deployment
        captured["texts"] = texts
        return [[0.1, 0.2]]

    def fail_vllm(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("vLLM path should not be used for azure backend")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "my-embed-deploy")
    monkeypatch.setattr(llm, "_call_azure_embeddings", fake_azure)
    monkeypatch.setattr(llm, "_call_openai_embeddings", fail_vllm)

    out = llm.embed(["hello"])
    assert out == [[0.1, 0.2]]
    assert captured["deployment"] == "my-embed-deploy"
    assert captured["texts"] == ["hello"]


def test_embed_azure_model_prefix_strips_and_overrides_env(monkeypatch):
    captured = {}

    def fake_azure(texts, deployment, timeout):
        captured["deployment"] = deployment
        return [[1.0]]

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "env-default")
    monkeypatch.setattr(llm, "_call_azure_embeddings", fake_azure)

    llm.embed(["x"], model="azure/explicit-deploy")
    # "azure/" prefix strips to the explicit deployment, overriding the env default
    assert captured["deployment"] == "explicit-deploy"


def test_embed_defaults_to_vllm(monkeypatch):
    captured = {}

    def fake_vllm(texts, model, base_url, api_key, timeout):
        captured["model"] = model
        captured["base_url"] = base_url
        return [[0.0]]

    def fail_azure(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("azure path should not be used for vllm backend")

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("VLLM_EMBEDDING_ENDPOINT", "http://localhost:8003/v1")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "cl-nagoya/ruri-v3-310m")
    monkeypatch.setattr(llm, "_call_openai_embeddings", fake_vllm)
    monkeypatch.setattr(llm, "_call_azure_embeddings", fail_azure)

    llm.embed(["y"])
    assert captured["model"] == "cl-nagoya/ruri-v3-310m"
    assert captured["base_url"] == "http://localhost:8003/v1"
