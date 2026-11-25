"""LiteLLM gateway providing response caching, ordered key pool failover, and failure simulation."""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm import Router

from app.config import KeyConfig, Settings

logger = logging.getLogger(__name__)


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails for a request."""


@dataclass(frozen=True)
class GatewayResult:
    """Normalized LLM response and per-request telemetry."""

    reply: str
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header (e.g. 'key_1=429,key_2=500' or '*=429')."""
    if not header_value:
        return {}
    parsed: Dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if "=" not in pair:
            continue
        key_id, status_str = pair.split("=", 1)
        key_id = key_id.strip()
        status_str = status_str.strip()
        if not key_id or not status_str.isdigit():
            continue
        status_code = int(status_str)
        if status_code in VALID_SIMULATED_STATUSES:
            parsed[key_id] = status_code
    return parsed


def _build_simulated_exception(status_code: int, key_cfg: KeyConfig) -> Exception:
    """Construct a LiteLLM-compatible HTTP error for simulated key failures."""
    provider = key_cfg.model.split("/", 1)[0] if "/" in key_cfg.model else "unknown"
    msg = f"Simulated provider failure ({status_code}) for {key_cfg.key_id}"
    if status_code == 429:
        return litellm.RateLimitError(
            message=msg, llm_provider=provider, model=key_cfg.model
        )
    if status_code == 502:
        return litellm.BadGatewayError(
            message=msg, llm_provider=provider, model=key_cfg.model
        )
    if status_code == 503:
        return litellm.ServiceUnavailableError(
            message=msg, llm_provider=provider, model=key_cfg.model
        )
    return litellm.InternalServerError(
        message=msg, llm_provider=provider, model=key_cfg.model
    )


def is_failover_error(exc: Exception) -> bool:
    """Return True if the exception represents rate limit, quota exhaustion, or 5xx/transient error."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True

    retryable_types = (
        litellm.RateLimitError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
        litellm.Timeout,
        litellm.APIConnectionError,
    )
    if isinstance(exc, retryable_types):
        return True

    text = str(exc).lower()
    keywords = ("429", "rate limit", "rate_limit", "quota", "resource_exhausted", "500", "502", "503", "504")
    return any(kw in text for kw in keywords)


def _extract_reply_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return str(content).strip() if content is not None else ""


def _extract_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0

    return int(prompt_tokens), int(completion_tokens)


def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        return bool(hidden.get("cache_hit"))
    return False


class LLMGateway:
    """Manages LiteLLM built-in response caching and ordered multi-key failover."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys: List[KeyConfig] = settings.llm_keys
        self.cache_enabled = True

        # Enable LiteLLM's built-in in-memory response cache.
        litellm.cache = litellm.Cache(type="local")
        litellm.suppress_debug_info = True

        # Register deployments with LiteLLM Router for clean multi-key configuration.
        model_list = [
            {
                "model_name": "chat-pool",
                "litellm_params": {
                    "model": key_cfg.model,
                    "api_key": key_cfg.api_key,
                },
                "model_info": {
                    "id": key_cfg.key_id,
                },
            }
            for key_cfg in self.keys
        ]
        self.router = Router(model_list=model_list) if model_list else None

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        """Execute chat completion with LiteLLM caching and ordered key failover."""
        if not self.keys:
            raise AllProvidersUnavailableError("No LLM API keys configured")

        simulated_failures = (
            parse_simulate_failure_header(simulate_failure_header)
            if self.settings.enable_failure_simulation
            else {}
        )
        bypass_cache = bool(simulated_failures)

        start = time.perf_counter()
        fallback_attempts = 0
        last_error: Optional[Exception] = None

        for key_cfg in self.keys:
            simulated_status = simulated_failures.get(key_cfg.key_id) or simulated_failures.get("*")
            if simulated_status is not None:
                last_error = _build_simulated_exception(simulated_status, key_cfg)
                logger.warning(
                    "Simulated failure status=%s on %s", simulated_status, key_cfg.key_id
                )
                fallback_attempts += 1
                continue

            try:
                completion_kwargs: Dict[str, Any] = {
                    "model": key_cfg.model,
                    "messages": messages,
                    "api_key": key_cfg.api_key,
                    "caching": not bypass_cache,
                    "num_retries": 0,
                }
                response = litellm.completion(**completion_kwargs)
                reply_text = _extract_reply_text(response)
                if not reply_text:
                    raise litellm.InternalServerError(
                        message=f"Empty completion returned by {key_cfg.key_id}",
                        llm_provider="litellm",
                        model=key_cfg.model,
                    )

                latency_ms = max(0, int(round((time.perf_counter() - start) * 1000)))
                input_tokens, output_tokens = _extract_tokens(response)
                cache_hit = (not bypass_cache) and _is_cache_hit(response)

                return GatewayResult(
                    reply=reply_text,
                    resolution_source="LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                    model=key_cfg.model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None if cache_hit else key_cfg.key_id,
                    fallback_attempts=0 if cache_hit else fallback_attempts,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Key %s failed (%s): %s",
                    key_cfg.key_id,
                    type(exc).__name__,
                    str(exc),
                )
                if is_failover_error(exc):
                    fallback_attempts += 1
                    continue
                # Fail over on unexpected provider errors too so single bad key never breaks chat.
                fallback_attempts += 1
                continue

        logger.error("All %d configured keys failed. Last error: %s", len(self.keys), last_error)
        raise AllProvidersUnavailableError("all providers unavailable")
