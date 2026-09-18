from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .agent import OpenRouterError
from .channels import (
    ChannelRequest,
    ChannelService,
    teams_question,
    teams_reply,
)
from .config import (
    API_ALLOWED_HOSTS,
    APP_ENV,
    CORS_ORIGINS,
    METRICS_TOKEN,
    RATE_LIMIT_PER_MINUTE,
)
from .logging_config import configure_logging
from .memory_cleanup import cleanup_old_conversations
from .security import (
    RequestSizeLimitMiddleware,
    verify_api_key,
    verify_teams_signature,
)
from .store import IndexCompatibilityError, store_exists

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (in-process; swap get_remote_address for Redis in production)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
_rate_limit_string = f"{RATE_LIMIT_PER_MINUTE}/minute"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


# Validate messages sent by web clients.
class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)


# Describe one evidence citation in an API response.
class Citation(BaseModel):
    source: str | None = None
    relative_path: str | None = None
    page: int | None = None
    chunk: int | None = None
    retrieval_score: float | None = None


# Describe RAG timing and cache performance.
class Metrics(BaseModel):
    retrieval_time_s: float | None = None
    generation_time_s: float | None = None
    total_time_s: float
    cache_hit: bool


# Define the structured response returned to web clients.
class ChatResponse(BaseModel):
    channel: Literal["web"]
    conversation_id: str | None = None
    user_id: str | None = None
    history_messages_used: int
    query: str
    answer: str
    citations: dict[str, Citation]
    metrics: Metrics


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


# Retrieve the application-scoped channel service.
def get_channel_service(request: Request) -> ChannelService:
    return request.app.state.channel_service


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


# Build and configure the FastAPI application.
def create_app(channel_service: ChannelService | None = None) -> FastAPI:
    # Configure structured logging as early as possible.
    configure_logging(APP_ENV)

    # Initialize shared application services for the server lifetime.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.channel_service = channel_service or ChannelService()

        # Start daily background cleanup for old conversation rows.
        try:
            from apscheduler.schedulers.asyncio import (
                AsyncIOScheduler,  # type: ignore[import-untyped]
            )

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                cleanup_old_conversations,
                trigger="interval",
                hours=24,
                id="conversation_cleanup",
                replace_existing=True,
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info("APScheduler started — conversation cleanup every 24 h")
        except ImportError:
            logger.warning("apscheduler not installed; conversation cleanup disabled")

        yield

        # Graceful shutdown
        if hasattr(app.state, "scheduler"):
            app.state.scheduler.shutdown(wait=False)

    app = FastAPI(
        title="Machine Learning AI Assistant",
        version="0.3.0",
        description="Shared RAG endpoints for web clients and Microsoft Teams.",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Middleware stack (outermost → innermost)
    # -----------------------------------------------------------------------

    # Rate-limiter state (must be set before any route uses @limiter.limit).
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))

    # Body-size cap — applied before routing to prevent memory exhaustion.
    app.add_middleware(RequestSizeLimitMiddleware)

    # CORS — only add origins if explicitly configured (empty = no CORS headers).
    if CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Host header restriction (empty list = accept any host for local dev).
    if API_ALLOWED_HOSTS:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=API_ALLOWED_HOSTS)

    # -----------------------------------------------------------------------
    # Operations endpoints (no auth, safe for load-balancer probes)
    # -----------------------------------------------------------------------

    # Redirect root to interactive API docs.
    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # Report whether the API process is running.
    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Report whether the document index is ready for queries.
    @app.get("/ready", tags=["operations"])
    async def ready() -> dict[str, str]:
        if not store_exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The document index is not available. Run ingestion first.",
            )
        return {"status": "ready"}

    # Return API version information.
    @app.get("/api/version", tags=["operations"])
    async def api_version() -> dict[str, str]:
        return {"version": "0.3.0", "api_version": "v1"}

    # Prometheus metrics endpoint — gated by METRICS_TOKEN when set.
    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    async def metrics(request: Request) -> Any:
        if METRICS_TOKEN:
            auth = request.headers.get("Authorization", "")
            import secrets as _s

            if not auth.startswith("Bearer ") or not _s.compare_digest(auth[7:], METRICS_TOKEN):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        try:
            from prometheus_client import (  # type: ignore[import-untyped]
                CONTENT_TYPE_LATEST,
                generate_latest,
            )
            from starlette.responses import Response as _Resp

            return _Resp(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            return {"detail": "prometheus_client not installed"}

    # -----------------------------------------------------------------------
    # Channel endpoints (auth + rate-limited)
    # -----------------------------------------------------------------------

    # Answer a message from a web application.
    @app.post(
        "/api/v1/channels/web/messages",
        response_model=ChatResponse,
        tags=["channels"],
    )
    @limiter.limit(_rate_limit_string)
    async def web_message(
        request: Request,
        body: ChatRequest,
        service: Annotated[ChannelService, Depends(get_channel_service)],
        _auth: Annotated[None, Security(verify_api_key)],
    ) -> dict[str, Any]:
        return await _answer(
            service,
            ChannelRequest(
                channel="web",
                question=body.message,
                conversation_id=body.conversation_id,
                user_id=body.user_id,
            ),
        )

    # Answer or ignore an incoming Teams activity.
    @app.post("/api/v1/channels/teams/messages", tags=["channels"])
    @limiter.limit(_rate_limit_string)
    async def teams_message(
        request: Request,
        service: Annotated[ChannelService, Depends(get_channel_service)],
        _body: Annotated[bytes, Depends(verify_teams_signature)],
    ) -> dict[str, Any]:
        # Body was verified and cached on request.state by the dependency.
        raw = getattr(request.state, "raw_body", b"{}")
        try:
            activity: dict[str, Any] = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Teams activity body is not valid JSON.",
            ) from exc

        activity_type = activity.get("type")
        if activity_type != "message":
            return {"status": "ignored", "activity_type": activity_type}
        question = teams_question(activity)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The Teams message contains no question.",
            )
        result = await _answer(
            service,
            ChannelRequest(
                channel="teams",
                question=question,
                conversation_id=(activity.get("conversation") or {}).get("id"),
                user_id=(activity.get("from") or {}).get("id"),
            ),
        )
        return teams_reply(activity, result)

    return app


# ---------------------------------------------------------------------------
# Shared answer helper
# ---------------------------------------------------------------------------


# Run synchronous retrieval and generation outside the event loop.
async def _answer(
    service: ChannelService,
    request: ChannelRequest,
) -> dict[str, Any]:
    try:
        # Retrieval and generation are synchronous, so keep them off the event loop.
        return await run_in_threadpool(service.ask, request)
    except (IndexCompatibilityError, OpenRouterError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


app = create_app()
