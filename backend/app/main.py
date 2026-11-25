"""FastAPI application exposing /health, /chat, and /sessions/{session_id}/messages."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.db import Database
from app.llm import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = custom_settings or load_settings()
    db = Database(settings.sqlite_path)
    gateway = LLMGateway(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        logger.info(
            "Backend started: keys=%d sqlite=%s failure_sim=%s",
            len(settings.llm_keys),
            settings.sqlite_path,
            settings.enable_failure_simulation,
        )
        yield

    app = FastAPI(
        title="Lyftr AI LLM Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.db = db
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        if not db.check_health():
            raise HTTPException(status_code=503, detail="database unavailable")
        if len(settings.llm_keys) < 1:
            raise HTTPException(status_code=503, detail="no api keys configured")
        return HealthResponse(
            status="ok",
            keys_configured=len(settings.llm_keys),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def post_chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        saved_system_prompt, history = db.get_session_context(payload.session_id)

        requested_system_prompt: Optional[str] = None
        if payload.system_prompt is not None and payload.system_prompt.strip():
            requested_system_prompt = payload.system_prompt.strip()

        effective_system_prompt = (
            requested_system_prompt
            or saved_system_prompt
            or settings.default_system_prompt
        )

        llm_messages = [
            {"role": "system", "content": effective_system_prompt},
            *history,
            {"role": "user", "content": payload.message},
        ]

        try:
            result = gateway.generate_reply(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        db.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            metadata=metadata.model_dump(),
            system_prompt=requested_system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    def get_session_messages(session_id: str) -> SessionMessagesResponse:
        messages = db.get_session_messages(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
