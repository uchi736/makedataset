import pytest


@pytest.fixture(autouse=True)
def _no_real_reranker(monkeypatch):
    """単体テストは実リランカー (:8006) に触れない。

    どれかのテストが rageval.cli を import すると load_dotenv() が走り、
    RERANK_ENDPOINT が入って rag_verify_batch が本物の API を呼んでしまう
    (2026-08-05: スイート実行時のみ3件が順序依存で落ちた)。リランカーを
    使うテストは fake を明示注入する。
    """
    monkeypatch.delenv("RERANK_ENDPOINT", raising=False)
    monkeypatch.delenv("RERANK_MODEL", raising=False)
