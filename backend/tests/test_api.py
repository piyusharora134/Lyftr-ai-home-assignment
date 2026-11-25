"""Contract and integration tests for the Lyftr AI LLM Gateway backend."""

import time
from typing import Any, Dict, List
from unittest.mock import patch

from fastapi.testclient import TestClient
import litellm
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM

from app.config import KeyConfig, Settings
from app.main import create_app


class FakeGeminiBackend:
    """Deterministic stub for VertexLLM.completion so real LiteLLM cache & routing run end-to-end."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail_by_api_key: Dict[str, Exception] = {}
        self.delay_seconds: float = 0.0

    def __call__(
        self,
        model: str,
        messages: list,
        model_response: litellm.ModelResponse,
        *args,
        gemini_api_key: str | None = None,
        **kwargs,
    ) -> litellm.ModelResponse:
        api_key = gemini_api_key or ""
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "api_key": api_key,
            }
        )

        if api_key in self.fail_by_api_key:
            raise self.fail_by_api_key[api_key]

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        last_user = messages[-1]["content"] if messages else ""
        key_num = api_key.split("-")[-1] if api_key else "?"
        prompt_tokens = sum(len(m["content"].split()) + 4 for m in messages)
        reply = f"Answer({key_num}): {last_user}"
        completion_tokens = len(reply.split())

        return litellm.ModelResponse(
            id="chatcmpl-test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            model=model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


def _make_settings(
    tmp_path,
    enable_sim: bool = True,
    db_name: str = "test_app.db",
) -> Settings:
    db_file = tmp_path / db_name
    return Settings(
        llm_keys=[
            KeyConfig(key_id="key_1", api_key="secret-1", model="gemini/gemini-2.0-flash"),
            KeyConfig(key_id="key_2", api_key="secret-2", model="gemini/gemini-2.0-flash"),
        ],
        database_url=f"sqlite:///{db_file}",
        sqlite_path=str(db_file),
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_sim,
        default_system_prompt="You are a helpful AI assistant.",
    )


def test_health_endpoint(tmp_path):
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {
            "status": "ok",
            "keys_configured": 2,
            "cache_enabled": True,
        }


def test_validation_errors_422_and_not_saved(tmp_path):
    fake = FakeGeminiBackend()
    app = create_app(_make_settings(tmp_path))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app) as client:
        # Whitespace-only message -> 422
        r1 = client.post("/chat", json={"session_id": "s-bad-1", "message": "   \n\t  "})
        assert r1.status_code == 422
        assert "detail" in r1.json()

        # Empty session_id -> 422
        r2 = client.post("/chat", json={"session_id": "", "message": "hello"})
        assert r2.status_code == 422
        assert "detail" in r2.json()

        # Over 4000 chars after trim -> 422
        r3 = client.post("/chat", json={"session_id": "s-bad-2", "message": "a" * 4001})
        assert r3.status_code == 422
        assert "detail" in r3.json()

        # Malformed JSON -> 422
        r4 = client.post(
            "/chat",
            content="{not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r4.status_code == 422
        assert "detail" in r4.json()

        # Exactly 4000 chars with surrounding whitespace -> 200
        r5 = client.post(
            "/chat",
            json={"session_id": "s-ok-4000", "message": "  " + ("x" * 4000) + "  "},
        )
        assert r5.status_code == 200

        # Failed requests must not create session history
        hist = client.get("/sessions/s-bad-1/messages")
        assert hist.status_code == 404
        assert hist.json() == {"detail": "session not found"}


def test_litellm_cache_hit_across_sessions_and_per_request_tokens(tmp_path):
    fake = FakeGeminiBackend()
    fake.delay_seconds = 0.06
    app = create_app(_make_settings(tmp_path))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app) as client:
        # 1. First request -> LIVE_INFERENCE
        r1 = client.post(
            "/chat",
            json={"session_id": "s-cache-1", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["session_id"] == "s-cache-1"
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["model"] == "gemini/gemini-2.0-flash"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["input_tokens"] > 0
        assert d1["metadata"]["output_tokens"] > 0
        assert len(fake.calls) == 1

        # 2. Same first message in a NEW session -> LLM_CACHE, faster, key_id=None
        r2 = client.post(
            "/chat",
            json={"session_id": "s-cache-2", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["session_id"] == "s-cache-2"
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["key_id"] is None
        assert d2["metadata"]["fallback_attempts"] == 0
        assert d2["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
        assert d2["metadata"]["output_tokens"] == d1["metadata"]["output_tokens"]
        assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]
        assert len(fake.calls) == 1  # Provider was not called on cache hit

        # 3. Different message of equal word length in new session -> LIVE_INFERENCE, same token count
        r3 = client.post(
            "/chat",
            json={"session_id": "s-cache-3", "message": "What is the capital of Brazil?"},
        )
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d3["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
        assert len(fake.calls) == 2


def test_multiturn_state_isolation_system_prompt_and_sqlite_persistence(tmp_path):
    fake = FakeGeminiBackend()
    app1 = create_app(_make_settings(tmp_path, db_name="persist.db"))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app1) as client:
        t1 = client.post(
            "/chat",
            json={
                "session_id": "s-multi-1",
                "message": "Remember BLUEBIRD",
                "system_prompt": "You are a pirate.",
            },
        )
        assert t1.status_code == 200

        # Turn 2 omits system_prompt; backend must keep session's system prompt + Turn 1 history
        t2 = client.post(
            "/chat",
            json={"session_id": "s-multi-1", "message": "What word did I say?"},
        )
        assert t2.status_code == 200
        assert len(fake.calls) == 2
        turn2_msgs = fake.calls[1]["messages"]
        assert turn2_msgs[0] == {"role": "system", "content": "You are a pirate."}
        assert turn2_msgs[1] == {"role": "user", "content": "Remember BLUEBIRD"}
        assert turn2_msgs[2]["role"] == "assistant"
        assert turn2_msgs[3] == {"role": "user", "content": "What word did I say?"}

        # Token count grows on turn 2 because history is included
        assert t2.json()["metadata"]["input_tokens"] > t1.json()["metadata"]["input_tokens"]

        # Separate session must not see s-multi-1 history or custom system prompt
        other = client.post(
            "/chat",
            json={"session_id": "s-other-1", "message": "Hello"},
        )
        assert other.status_code == 200
        other_msgs = fake.calls[2]["messages"]
        assert len(other_msgs) == 2
        assert other_msgs[0] == {"role": "system", "content": "You are a helpful AI assistant."}

    # Simulate container restart by creating a new app instance pointing at same SQLite DB
    app2 = create_app(_make_settings(tmp_path, db_name="persist.db"))
    with TestClient(app2) as client:
        hist = client.get("/sessions/s-multi-1/messages")
        assert hist.status_code == 200
        body = hist.json()
        assert body["session_id"] == "s-multi-1"
        assert len(body["messages"]) == 4
        assert [m["role"] for m in body["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        # System prompt must never appear in GET /sessions/{id}/messages
        assert all(m["role"] != "system" for m in body["messages"])
        assert "metadata" not in body["messages"][0] or body["messages"][0]["metadata"] is None
        assert body["messages"][1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_real_provider_failover_on_429_and_5xx(tmp_path):
    fake = FakeGeminiBackend()
    fake.fail_by_api_key["secret-1"] = litellm.RateLimitError(
        message="quota exceeded 429", llm_provider="gemini", model="gemini/gemini-2.0-flash"
    )
    app = create_app(_make_settings(tmp_path))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app) as client:
        res = client.post(
            "/chat",
            json={"session_id": "s-real-fail", "message": "Test real failover"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["metadata"]["key_id"] == "key_2"
        assert data["metadata"]["fallback_attempts"] == 1
        assert data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert "secret-1" not in data["reply"]
        assert "secret-2" not in data["reply"]


def test_x_simulate_failure_hook_and_503(tmp_path):
    fake = FakeGeminiBackend()
    app = create_app(_make_settings(tmp_path, enable_sim=True))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app) as client:
        # key_1=429 -> falls back to key_2
        r1 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=429"},
            json={"session_id": "s-sim-1", "message": "Simulated 429"},
        )
        assert r1.status_code == 200
        assert r1.json()["metadata"]["key_id"] == "key_2"
        assert r1.json()["metadata"]["fallback_attempts"] == 1

        # key_1=500 -> falls back to key_2
        r2 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=500"},
            json={"session_id": "s-sim-2", "message": "Simulated 500"},
        )
        assert r2.status_code == 200
        assert r2.json()["metadata"]["key_id"] == "key_2"
        assert r2.json()["metadata"]["fallback_attempts"] == 1

        # Subsequent normal request still uses key_1 (simulation did not poison key_1)
        r3 = client.post(
            "/chat",
            json={"session_id": "s-sim-3", "message": "Normal after simulation"},
        )
        assert r3.status_code == 200
        assert r3.json()["metadata"]["key_id"] == "key_1"
        assert r3.json()["metadata"]["fallback_attempts"] == 0

        # *=429 -> 503 and nothing added to history
        r4 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-sim-all", "message": "All fail"},
        )
        assert r4.status_code == 503
        assert r4.json() == {"detail": "all providers unavailable"}

        hist = client.get("/sessions/s-sim-all/messages")
        assert hist.status_code == 404
        assert hist.json() == {"detail": "session not found"}


def test_simulate_failure_ignored_when_disabled(tmp_path):
    fake = FakeGeminiBackend()
    app = create_app(_make_settings(tmp_path, enable_sim=False))

    with patch.object(VertexLLM, "completion", new=fake), TestClient(app) as client:
        res = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-sim-off", "message": "Should ignore header"},
        )
        assert res.status_code == 200
        assert res.json()["metadata"]["key_id"] == "key_1"
        assert res.json()["metadata"]["fallback_attempts"] == 0
