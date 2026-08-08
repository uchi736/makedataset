"""Tests for llm backend selection (vLLM / Azure) and model family kwargs."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_parse_json_tolerates_stray_leading_brace():
    """回帰: DeepSeek が JSON の前に迷子の '{' を吐いても抽出できること。"""
    from rageval.llm import _parse_json
    out = _parse_json('{\n{"entailed": false, "reason": "理由"}')
    assert out == {"entailed": False, "reason": "理由"}


# ---------- 提供元の固定 (LLM_PROVIDER) ----------

def test_resolve_backend_without_provider_uses_model_name(monkeypatch):
    """LLM_PROVIDER 未設定なら従来どおりモデル名と明示引数で決まる。"""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    assert llm._resolve_backend("gpt-4.1-mini", None) == "azure"
    assert llm._resolve_backend("azure/gpt-4.1-mini", None) == "azure"
    assert llm._resolve_backend("claude-sonnet-4", None) == "anthropic"
    assert llm._resolve_backend("google/gemma-4-26B-A4B-it", None) == "vllm"
    assert llm._resolve_backend("gpt-4.1-mini", "vllm") == "vllm"


def test_llm_provider_vllm_blocks_azure_deployment_name(monkeypatch):
    """回帰: オンプレ固定のつもりでも配備名一致でクラウドへ発呼していた。"""
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER=vllm"):
        llm._resolve_backend("gpt-4.1-mini", None)
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER=vllm"):
        llm._resolve_backend("azure/anything", None)


def test_llm_provider_vllm_blocks_anthropic_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER=vllm"):
        llm._resolve_backend("claude-sonnet-4", None)


def test_llm_provider_conflicting_explicit_backend_raises(monkeypatch):
    """設定と呼び出しが食い違うとき、どちらを無視したかを黙って決めない。"""
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    with pytest.raises(llm.LLMError, match="食い違"):
        llm._resolve_backend("google/gemma-4-26B-A4B-it", "azure")


def test_llm_provider_vllm_allows_onprem_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    assert llm._resolve_backend("google/gemma-4-26B-A4B-it", None) == "vllm"
    assert llm._resolve_backend("google/gemma-4-26B-A4B-it", "vllm") == "vllm"


def test_llm_provider_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  VLLM ")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER=vllm"):
        llm._resolve_backend("gpt-4.1-mini", None)


def test_llm_provider_azure_overrides_model_name(monkeypatch):
    """クラウドを明示的に選んだときは最優先で通す (締め金具は vllm 側だけ)。"""
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
    assert llm._resolve_backend("google/gemma-4-26B-A4B-it", None) == "azure"


def test_llm_provider_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER"):
        llm._resolve_backend("google/gemma-4-26B-A4B-it", None)


def test_generate_refuses_cloud_call_when_pinned_to_vllm(monkeypatch):
    def fail(*a, **k):  # pragma: no cover - 呼ばれてはならない
        raise AssertionError("LLM_PROVIDER=vllm でクラウド経路を呼んだ")

    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    monkeypatch.setattr(llm, "_call_azure_openai", fail)
    monkeypatch.setattr(llm, "_call_anthropic", fail)

    with pytest.raises(llm.LLMError):
        llm.generate("hello", model="gpt-4.1-mini")
    with pytest.raises(llm.LLMError):
        llm.generate("hello", model="claude-sonnet-4")


def test_embedding_backend_ignores_llm_provider(monkeypatch):
    """埋め込み側は EMBEDDING_PROVIDER のみで決まる (既存挙動を変えない)。"""
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert llm._resolve_embedding_backend(None, None) == "vllm"


# ---------- モデル系統ごとの追加引数 ----------

def test_family_qwen_thinking_off_by_default():
    kwargs = {"response_format": {"type": "json_object"}}
    llm.apply_model_family_kwargs(kwargs, "Qwen/Qwen3-32B")
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["response_format"] == {"type": "json_object"}


def test_family_qwen_thinking_on():
    kwargs: dict = {}
    llm.apply_model_family_kwargs(kwargs, "Qwen/Qwen3-32B", allow_thinking=True)
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_family_deepseek_drops_response_format_and_sets_effort():
    kwargs = {"response_format": {"type": "json_object"}}
    llm.apply_model_family_kwargs(kwargs, "deepseek-ai/DeepSeek-V4")
    assert "response_format" not in kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_family_deepseek_qwen_accumulates_extra_body():
    """回帰: 両方の語を含む配信名で qwen 側が reasoning_effort を消していた。

    消えると思考が max_tokens を食い潰して本文が空になり、判定が全件未判定に落ちる。
    """
    kwargs = {"response_format": {"type": "json_object"}}
    llm.apply_model_family_kwargs(
        kwargs, "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", allow_thinking=True,
    )
    assert "response_format" not in kwargs
    assert kwargs["extra_body"]["reasoning_effort"] == "low"
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_family_unknown_model_gets_no_extra_body():
    kwargs = {"response_format": {"type": "json_object"}}
    llm.apply_model_family_kwargs(kwargs, "google/gemma-4-26B-A4B-it")
    assert "extra_body" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_allow_thinking_warns_when_no_family_supports_it():
    """回帰: 判定を Nemotron へ移すと思考指定が例外も警告も出さず消えていた。"""
    kwargs: dict = {}
    with pytest.warns(RuntimeWarning, match="allow_thinking"):
        llm.apply_model_family_kwargs(
            kwargs, "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5", allow_thinking=True,
        )
    assert "extra_body" not in kwargs


def test_no_warning_when_thinking_not_requested(recwarn):
    llm.apply_model_family_kwargs({}, "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5")
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_matching_model_families_is_order_stable():
    names = [f.name for f in llm.matching_model_families(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")]
    assert names == ["deepseek", "qwen"]
    assert llm.matching_model_families("google/gemma-4-26B-A4B-it") == []


def test_call_openai_compatible_sends_family_kwargs(monkeypatch):
    """表が実際に呼び出し引数へ届いていること (配線の凍結)。"""
    import openai

    captured: dict = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    out = llm._call_openai_compatible(
        prompt="p",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=64,
        json_mode=True,
        system=None,
        timeout=5.0,
        allow_thinking=True,
    )
    assert out == "ok"
    assert "response_format" not in captured
    assert captured["extra_body"]["reasoning_effort"] == "low"
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_generate_warns_when_thinking_requested_on_azure(monkeypatch):
    """azure 経路は思考指定を中継しない。黙って捨てると採否率の取り違えになる。"""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    monkeypatch.setattr(llm, "_call_azure_openai", lambda **kw: "ok")

    with pytest.warns(RuntimeWarning, match="allow_thinking"):
        assert llm.generate(
            "p", model="gpt-4.1-mini", allow_thinking=True) == "ok"
