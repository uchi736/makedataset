"""FastAPI アプリ: /api/* + ビルド済み React UI (webui/dist) の配信。

起動:
    uv run rageval web                     # http://127.0.0.1:8765
    uv run rageval web --port 9000 --data-root data
開発時 (フロントを Vite dev server で動かす場合):
    uv run rageval web --no-open           # API だけ使う
    cd webui && npm run dev                # :5173 から /api をプロキシ
"""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .labels import build_labels
from .service import ServiceError, list_sources, load_items, resolve_evidence, save_review

# webui/dist (リポジトリ直下)。src/rageval/webapi/app.py から3階層上がる。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DIST_DIR = _PROJECT_ROOT / "webui" / "dist"

app = FastAPI(title="rageval web", docs_url="/api/docs", openapi_url="/api/openapi.json")

# Vite dev server (npm run dev) からの呼び出しを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 起動時に run_server() が設定する
app.state.data_root = Path("data")
app.state.reviewer = "unknown"


def _data_root() -> Path:
    return Path(app.state.data_root)


@app.exception_handler(ServiceError)
async def _service_error_handler(_, exc: ServiceError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------- API ----------------

@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    """データセット一覧・ラベル定義・既定値。UI が最初に1回読む。"""
    return {
        "sources": list_sources(_data_root()),
        "labels": build_labels(),
        "defaults": {"reviewer": app.state.reviewer},
    }


@app.get("/api/items")
def api_items(path: str, track: str = "all") -> dict[str, Any]:
    """QA JSONL の読み込み。path は data/ からの相対 (dir / file / glob)。"""
    if track not in ("all", "general", "kg_poc"):
        raise HTTPException(status_code=400, detail=f"不正な track: {track}")
    return load_items(_data_root(), path, track)


class EvidenceRequest(BaseModel):
    # "auto" = data/ 配下の chunks* から引用が最も一致するものを自動選択
    chunks_dir: str = "auto"
    rationale: list[dict[str, Any]]


@app.post("/api/evidence")
def api_evidence(req: EvidenceRequest) -> dict[str, Any]:
    """根拠チャンクの解決 + 逐語一致チェック + ハイライト範囲。"""
    return resolve_evidence(_data_root(), req.chunks_dir, req.rationale)


class ReviewRequest(BaseModel):
    path: str                 # 読み込み中のデータセット (dir / file / glob)
    source: str               # 対象 QA が属するファイル名
    qa_id: str
    status: str               # accepted / edited / rejected / pending
    reviewer: str = "unknown"
    question: Optional[str] = None
    answer: Optional[str] = None


@app.post("/api/review")
def api_review(req: ReviewRequest) -> dict[str, Any]:
    """レビュー判定を保存する。書き込み先は data/reviewed/<元ファイル名>。"""
    return save_review(
        data_root=_data_root(),
        rel=req.path,
        source_name=req.source,
        qa_id=req.qa_id,
        status=req.status,
        reviewer=req.reviewer,
        question=req.question,
        answer=req.answer,
    )


# ---------------- 静的ファイル (ビルド済み React) ----------------

if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="webui")
else:
    @app.get("/")
    def _no_build() -> FileResponse:  # pragma: no cover
        raise HTTPException(
            status_code=503,
            detail=(
                "webui/dist がありません。リポジトリ直下で "
                "`cd webui && npm install && npm run build` を実行してください。"
            ),
        )


# ---------------- 起動 ----------------

def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    data_root: Path = Path("data"),
    reviewer: str = "unknown",
    open_browser: bool = True,
) -> None:
    import uvicorn

    app.state.data_root = Path(data_root)
    app.state.reviewer = reviewer
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run_server()
