"""
FastAPI application – exposes the RAG pipeline over HTTP with
authentication, rate limiting, structured logging, and error handling.

Endpoints
---------
GET  /        – welcome message (public)
GET  /health  – health check with index status (public)
POST /ask     – ask a question, get an LLM-grounded answer (auth required)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

# Ensure the project root is on sys.path so absolute imports work when
# running with ``uvicorn api.main:app`` from the repo root.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.auth import verify_api_key
from api.logger import log_request, setup_logger
from llm.llm_client import generate_answer

# ── Logger ───────────────────────────────────────────────────────────────────
logger = setup_logger("api")

# ── Rate limiter ─────────────────────────────────────────────────────────────
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Engineering Knowledge Base",
    description="RAG-powered Q&A over internal engineering documentation.",
    version="0.2.0",
)
app.state.limiter = limiter


# ── Rate-limit error handler ────────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Max 10 requests/minute."},
    )


# ── Global exception handlers ───────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict:
    """Public welcome endpoint."""
    return {"message": "AI Engineering Knowledge Base API running"}


@app.get("/health")
def health() -> dict:
    """Health check — reports model name and whether the FAISS index is loaded."""
    index_path = Path(PROJECT_ROOT) / "vector_store" / "faiss_index" / "index.faiss"
    return {
        "status": "ok",
        "model": os.getenv("OLLAMA_MODEL", "mistral"),
        "index_loaded": index_path.exists(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.post("/ask", response_model=AskResponse)
@limiter.limit(RATE_LIMIT)
def ask_endpoint(
    req: AskRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> AskResponse:
    """Ask a question and receive an answer grounded in the knowledge base."""
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    try:
        answer = generate_answer(req.question)
        duration_ms = (time.perf_counter() - start) * 1000

        # Extract sources from the answer text (after "Sources:" line)
        sources: list[str] = []
        if "\nSources:" in answer:
            sources_text = answer.split("\nSources:")[-1].strip()
            sources = [s.strip() for s in sources_text.split("\n") if s.strip()]

        log_request(
            logger,
            endpoint="/ask",
            question=req.question,
            answer_length=len(answer),
            sources=sources,
            duration_ms=duration_ms,
            status="success",
            ip=client_ip,
        )
        return AskResponse(question=req.question, answer=answer)

    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        log_request(
            logger,
            endpoint="/ask",
            question=req.question,
            answer_length=0,
            sources=[],
            duration_ms=duration_ms,
            status="error",
            ip=client_ip,
            error=str(exc),
        )
        raise


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
