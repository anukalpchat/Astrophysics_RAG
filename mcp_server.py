"""
MCP server for the Astrophysics RAG pipeline.
Exposes two tools: search_papers (retrieval only) and ask_astrophysics (full RAG).
Transport: SSE over HTTP, with API key auth and per-key rate limiting.
"""

import os
import time
import uvicorn
from collections import defaultdict
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

from ai import AstrophysicsRAGChatbot

# --- Config ------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()

MCP_API_KEY = os.getenv("MCP_API_KEY", "")
RATE_LIMIT = 20  # max requests per 60-second window per key

# --- Middleware ---------------------------------------------------------------

class AuthMiddleware(BaseHTTPMiddleware):
    """Rejects any request missing the correct X-API-Key header."""

    async def dispatch(self, request: Request, call_next):
        if not MCP_API_KEY:
            # Fail closed: refuse all traffic if the server was started without a key
            return JSONResponse(
                {"error": "Server misconfigured: MCP_API_KEY environment variable is not set"},
                status_code=500,
            )
        if request.headers.get("X-API-Key", "") != MCP_API_KEY:
            return JSONResponse(
                {"error": "Unauthorized: provide a valid X-API-Key header"},
                status_code=401,
            )
        return await call_next(request)


_request_log: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter: max RATE_LIMIT requests per minute per API key."""

    async def dispatch(self, request: Request, call_next):
        # Use the API key as the identifier so limits are per-user, not per-IP.
        # Fall back to IP if somehow called without a key (auth middleware should block first).
        identifier = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        window_start = now - 60.0

        # Drop timestamps that have fallen outside the window
        _request_log[identifier] = [t for t in _request_log[identifier] if t > window_start]

        if len(_request_log[identifier]) >= RATE_LIMIT:
            return JSONResponse(
                {"error": f"Rate limit exceeded: max {RATE_LIMIT} requests per minute"},
                status_code=429,
            )

        _request_log[identifier].append(now)
        return await call_next(request)


# --- Chatbot (loaded once at startup) ----------------------------------------

chatbot = AstrophysicsRAGChatbot(
    index_path=str(BASE_DIR / "faiss_index" / "astrophysics.index"),
    metadata_path=str(BASE_DIR / "faiss_index" / "metadata.pkl"),
    chunks_file=str(BASE_DIR / "chunked_papers.json"),
)

# --- MCP tools ---------------------------------------------------------------

mcp = FastMCP("Astrophysics RAG")


@mcp.tool()
def search_papers(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the astrophysics paper index and return the most relevant text chunks.
    No LLM call is made — this is pure vector retrieval.

    Args:
        query: The search query or question.
        top_k: How many chunks to return (default 5).
    """
    context_texts, sources = chatbot.retrieve_context_with_text(query, top_k=top_k)
    return [
        {
            "text": text,
            "paper_id": src["paper_id"],
            "section": src["section"],
            "score": src["score"],
            "chunk_index": src["chunk_index"],
            "source_file": src["source_file"],
        }
        for text, src in zip(context_texts, sources)
    ]


@mcp.tool()
def ask_astrophysics(question: str, include_sources: bool = False) -> str:
    """
    Ask an astrophysics question. Retrieves relevant paper chunks, then calls
    the Groq LLM to produce a friendly, plain-language answer.

    Args:
        question: The astrophysics question to answer.
        include_sources: If True, appends a list of cited papers to the answer.
    """
    result = chatbot.ask(question, include_sources=include_sources)
    if result["success"]:
        return result["answer"]
    return f"Error: {result.get('error', 'Unknown error')}"


# --- Server entry point ------------------------------------------------------

if __name__ == "__main__":
    app = mcp.sse_app()
    # Middleware order: AuthMiddleware is outermost (checked first),
    # RateLimitMiddleware is inner (only reached after auth passes).
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    port = int(os.getenv("PORT", 8000))
    print(f"Starting Astrophysics RAG MCP server on http://0.0.0.0:{port}")
    print(f"SSE endpoint: http://0.0.0.0:{port}/sse")
    uvicorn.run(app, host="0.0.0.0", port=port)
