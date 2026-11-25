"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class KeyConfig:
    """Represents one API key deployment in the LiteLLM pool."""

    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the backend service."""

    llm_keys: List[KeyConfig]
    database_url: str
    sqlite_path: str
    allowed_origins: List[str]
    enable_failure_simulation: bool
    default_system_prompt: str


def resolve_sqlite_path(database_url: str) -> str:
    """Convert a SQLAlchemy-style sqlite:/// URL to a filesystem path for sqlite3."""
    url = (database_url or "").strip()
    if not url:
        return "/data/app.db"
    if url == "sqlite:///:memory:" or url == ":memory:":
        return ":memory:"
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        return path or "/data/app.db"
    return url


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def load_settings() -> Settings:
    """Load and validate configuration from process environment."""
    raw_keys = _parse_csv(os.getenv("LLM_API_KEYS", ""))
    default_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    raw_models = _parse_csv(os.getenv("LLM_MODELS", ""))

    llm_keys: List[KeyConfig] = []
    for idx, api_key in enumerate(raw_keys, start=1):
        if len(raw_models) == len(raw_keys) and raw_models:
            model_name = raw_models[idx - 1]
        elif len(raw_models) == 1:
            model_name = raw_models[0]
        else:
            model_name = default_model

        llm_keys.append(
            KeyConfig(
                key_id=f"key_{idx}",
                api_key=api_key,
                model=model_name,
            )
        )

    database_url = os.getenv("DATABASE_URL", "sqlite:////data/app.db")
    sqlite_path = resolve_sqlite_path(database_url)
    if sqlite_path != ":memory:":
        parent = Path(sqlite_path).parent
        if str(parent) and str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)

    raw_origins = _parse_csv(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )
    allowed_origins = raw_origins if raw_origins else ["http://localhost:3000"]

    enable_sim = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() == "true"
    default_prompt = (
        os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful AI assistant.").strip()
        or "You are a helpful AI assistant."
    )

    return Settings(
        llm_keys=llm_keys,
        database_url=database_url,
        sqlite_path=sqlite_path,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_sim,
        default_system_prompt=default_prompt,
    )
