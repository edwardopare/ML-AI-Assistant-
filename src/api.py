from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from .agent import OpenRouterError
from .channels import (
    ChannelRequest,
    ChannelService,
    teams_question,
    teams_reply,
)
from .store import IndexCompatibilityError, store_exists


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)


class Citation(BaseModel):
    source: str | None = None
    relative_path: str | None = None
    page: int | None = None
    chunk: int | None = None
    retrieval_score: float | None = None


class Metrics(BaseModel):
    retrieval_time_s: float | None = None
    generation_time_s: float | None = None
    total_time_s: float
    cache_hit: bool


class ChatResponse(BaseModel):
    channel: Literal["web"]
    conversation_id: str | None = None
    user_id: str | None = None
    query: str
    answer: str
    citations: dict[str, Citation]
    metrics: Metrics


def get_channel_service(request: Request) -> ChannelService:
    return request.app.state.channel_service


def create_app(channel_service: ChannelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.channel_service = channel_service or ChannelService()
        yield

    app = FastAPI(
        title="Machine Learning AI Assistant",
        version="1.0.0",
        description="Shared RAG endpoints for web clients and Microsoft Teams.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    async def ready() -> dict[str, str]:
        if not store_exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The document index is not available. Run ingestion first.",
            )
        return {"status": "ready"}

    @app.post(
        "/api/v1/channels/web/messages",
        response_model=ChatResponse,
        tags=["channels"],
    )
    async def web_message(
        body: ChatRequest,
        service: Annotated[ChannelService, Depends(get_channel_service)],
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

    @app.post("/api/v1/channels/teams/messages", tags=["channels"])
    async def teams_message(
        activity: dict[str, Any],
        service: Annotated[ChannelService, Depends(get_channel_service)],
    ) -> dict[str, Any]:
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
