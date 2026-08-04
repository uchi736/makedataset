"""LLM call wrappers for vLLM (OpenAI-compatible), Azure OpenAI, and Anthropic.

Single entry-point `generate(prompt, model, ...)` returns a string, or a
validated Pydantic object when `response_model=` is supplied.

Backend selection:
  - "anthropic": model starts with "claude" / "anthropic/", or backend="anthropic"
  - "azure":     backend="azure", or AZURE_OPENAI_ENDPOINT is set and model
                 matches the configured deployment / starts with "azure/"
  - "vllm":      default (OpenAI-compatible endpoint at VLLM_ENDPOINT)

Env vars:
  - VLLM_ENDPOINT (fallback: VLLM_BASE_URL), VLLM_API_KEY, VLLM_TIMEOUT
  - VLLM_MODEL (optional default generator)
  - VLLM_MODEL_ENDPOINTS (JSON object: モデル名 → base_url。生成と判定で別々の
    vLLM サーバを使うときの振り分け表。未登録モデルは VLLM_ENDPOINT に落ちる)
  - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
  - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
  - ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when an LLM call fails after retries or returns invalid output."""


def _http_client(timeout: float):
    """openai/httpx 用クライアント。`trust_env=False` でシステム/環境変数の
    プロキシ設定を一切無視する。

    背景: 一部の実行環境ではシステムプロキシが設定されており、httpx 既定
    (trust_env=True) だと POST (chat/completions・embeddings) がそのプロキシ
    経由になって握り潰され、GET (/models) は通るのに生成だけ ~47秒で
    APIConnectionError になる。LAN 上の vLLM もクラウドの Azure も対象なので、
    一律でプロキシを使わない。
    """
    import httpx

    return httpx.Client(trust_env=False, timeout=timeout)


def _strip_code_fence(text: str) -> str:
    """Extract JSON body from ```json ... ``` fences (anywhere in the text)."""
    stripped = text.strip()
    # Prefer fenced block; also accept fences with trailing commentary outside.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return stripped


def _extract_first_json(text: str) -> str | None:
    """Find the first balanced {...} object in text."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _parse_json(text: str) -> Any:
    candidates = [_strip_code_fence(text)]
    obj = _extract_first_json(text)
    if obj and obj not in candidates:
        candidates.append(obj)
    last_err: Exception | None = None
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError as e:
            last_err = e
    raise LLMError(f"Failed to parse JSON from LLM output: {last_err}\n---\n{text}")


# ---------- backend detection ----------

def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude") or model.startswith("anthropic/")


def _is_azure_model(model: str) -> bool:
    if model.startswith("azure/"):
        return True
    azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "")
    return bool(azure_deployment) and model == azure_deployment


def _resolve_backend(model: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if _is_anthropic_model(model):
        return "anthropic"
    if _is_azure_model(model):
        return "azure"
    return "vllm"


def _azure_deployment_from_model(model: str) -> str:
    if model.startswith("azure/"):
        return model.removeprefix("azure/")
    return model


def _vllm_model_endpoints() -> dict[str, str]:
    """VLLM_MODEL_ENDPOINTS (JSON object) を読む。壊れた JSON は黙って無視しない。"""
    raw = os.getenv("VLLM_MODEL_ENDPOINTS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON object (モデル名 → base_url) が必要")
        return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMError(f"VLLM_MODEL_ENDPOINTS が不正です: {e}\n---\n{raw}") from e


def _vllm_endpoint(model: Optional[str] = None) -> str:
    """モデル名に対応する vLLM base_url。振り分け表に無ければ既定エンドポイント。"""
    if model:
        mapped = _vllm_model_endpoints().get(model)
        if mapped:
            return mapped
    return os.getenv("VLLM_ENDPOINT") or os.getenv(
        "VLLM_BASE_URL", "http://localhost:8000/v1"
    )


def _vllm_timeout() -> float:
    raw = os.getenv("VLLM_TIMEOUT")
    return float(raw) if raw else 120.0


# ---------- backend implementations ----------

# 2026-08-03 縮小: 旧設定 (6回・待ち最大60秒) は Azure の 429 回復を待つための
# もので、オンプレ vLLM ではハングした1コールが最悪30分に化けてバッチが
# 「静かに止まる」原因になった (17時間無進捗の実測)。Azure は資格情報を
# 無効化済みなので、オンプレ前提の短い再試行に戻す。落ちたら層別カウントに
# 出る方が、粘って隠れるより良い。
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


@retry(**_RETRY_KWARGS)
def _call_openai_compatible(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    system: Optional[str],
    timeout: float,
    allow_thinking: bool = False,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout,
        http_client=_http_client(timeout),
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "qwen" in model.lower():
        # Qwen3 系は既定で思考 (reasoning) し、判定1コールに数十秒かかるうえ
        # max_tokens を思考が食い潰すと本文が空になる (2026-08-02 実測)。
        # 既定は思考オフ。ただし委任連鎖の適用範囲判定など、思考無しでは
        # 金看板素材を誤棄却する工程 (段1合成判定) だけ allow_thinking=True で
        # 呼び出し側が明示的に有効化する。Gemma 等のテンプレートはこの kwarg を
        # 参照しないため、他モデルには渡さない。
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": bool(allow_thinking)}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


@retry(**_RETRY_KWARGS)
def _call_azure_openai(
    prompt: str,
    deployment: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    system: Optional[str],
    timeout: float,
) -> str:
    from openai import AzureOpenAI

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not (endpoint and api_key):
        raise LLMError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout,
        http_client=_http_client(timeout),
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


@retry(**_RETRY_KWARGS)
def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system: Optional[str],
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


# ---------- public API ----------

def generate(
    prompt: str,
    model: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    # vLLM/Azureともmax_tokensは「上限」であり事前確保せず実出力分のみ課金される。
    # 過度に絞ると JSON 出力が truncate しやすくなるため、QA生成の余裕として
    # 8192 を既定とする。judge 系の短い出力は呼び出し側で上書きする。
    max_tokens: int = 8192,
    response_model: Optional[Type[T]] = None,
    force_json: bool = False,
    backend: Optional[str] = None,
    # 思考型モデル (Qwen3 系) の思考を許可するか。既定 False (速度優先)。
    # 委任連鎖の適用範囲判定など、思考無しで誤棄却が実測された工程だけ True。
    allow_thinking: bool = False,
) -> Any:
    """Generate text (or a validated Pydantic object) from an LLM.

    force_json=True で `response_format={"type": "json_object"}` を有効化
    (vLLM/Azure 共通: LLM 出力を JSON object に強制)。Pydantic スキーマ強制
    まではしないが、文字列の途中切断や fence 外コメント等は防げる。
    """
    backend = _resolve_backend(model, backend)
    json_mode = response_model is not None or force_json
    timeout = _vllm_timeout()

    if backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        text = _call_anthropic(
            prompt=prompt,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
    elif backend == "azure":
        deployment = _azure_deployment_from_model(model)
        text = _call_azure_openai(
            prompt=prompt,
            deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            system=system,
            timeout=timeout,
        )
    else:  # vllm / openai-compatible
        base_url = _vllm_endpoint(model)
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        text = _call_openai_compatible(
            prompt=prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            system=system,
            timeout=timeout,
            allow_thinking=allow_thinking,
        )

    if not (text or "").strip():
        # 思考型モデル (Qwen3 系) は reasoning が max_tokens を食い潰すと
        # 本文が空のまま正常応答を返す (2026-08-02 実測: max_tokens=512 で全滅)。
        # 空文字を呼び出し側に流すと「判定不能」が「不成立」等に化けるので
        # ここで止める。
        raise LLMError(
            f"LLM が空の本文を返しました (model={model}, max_tokens={max_tokens})。"
            "思考型モデルの思考が予算を使い切った可能性 — max_tokens を増やしてください"
        )

    if response_model is None:
        return text

    data = _parse_json(text)
    try:
        return response_model.model_validate(data)
    except ValidationError as e:
        raise LLMError(f"LLM output failed schema validation: {e}\n---\n{text}") from e


# ---------- embeddings ----------

def _resolve_embedding_backend(model: Optional[str], explicit: Optional[str]) -> str:
    """埋め込みバックエンド選択。chat の _resolve_backend と揃える。

    優先順位: 明示引数 > model の "azure/" 接頭辞 > EMBEDDING_PROVIDER 環境変数
    > 既定 "vllm"。
    """
    if explicit:
        return explicit
    if model and model.startswith("azure/"):
        return "azure"
    prov = os.getenv("EMBEDDING_PROVIDER")
    if prov:
        return prov.strip().lower()
    return "vllm"


@retry(**_RETRY_KWARGS)
def _call_openai_embeddings(
    texts: list[str], model: str, base_url: str, api_key: str, timeout: float,
) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout,
        http_client=_http_client(timeout),
    )
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


@retry(**_RETRY_KWARGS)
def _call_azure_embeddings(
    texts: list[str], deployment: str, timeout: float,
) -> list[list[float]]:
    from openai import AzureOpenAI

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not (endpoint and api_key):
        raise LLMError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set")

    client = AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
        timeout=timeout, http_client=_http_client(timeout),
    )
    resp = client.embeddings.create(model=deployment, input=texts)
    return [d.embedding for d in resp.data]


def embed(
    texts: list[str],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    backend: Optional[str] = None,
) -> list[list[float]]:
    """Embed a batch of texts (vLLM OpenAI-compatible or Azure OpenAI).

    Backend selection (_resolve_embedding_backend):
      - "azure": backend="azure", model が "azure/" 始まり、または
                 EMBEDDING_PROVIDER=azure。deployment は model(接頭辞除去) →
                 AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME の順で解決。
      - "vllm":  既定。VLLM_EMBEDDING_ENDPOINT / VLLM_EMBEDDING_MODEL を使用。

    base_url / api_key の明示指定は vLLM パスのみ有効 (Azure は環境変数から解決)。
    """
    timeout = timeout if timeout is not None else _vllm_timeout()
    backend = _resolve_embedding_backend(model, backend)

    if backend == "azure":
        deployment = _azure_deployment_from_model(model) if model else None
        deployment = deployment or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"
        )
        return _call_azure_embeddings(texts, deployment=deployment, timeout=timeout)

    # vllm / openai-compatible
    base_url = base_url or os.getenv("VLLM_EMBEDDING_ENDPOINT") or _vllm_endpoint()
    model = model or os.getenv("VLLM_EMBEDDING_MODEL", "cl-nagoya/ruri-v3-310m")
    api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
    return _call_openai_embeddings(
        texts, model=model, base_url=base_url, api_key=api_key, timeout=timeout,
    )
