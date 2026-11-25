# Lyftr AI — Full-Stack LLM Chatbot with Key Failover & Response Caching

A resilient, multi-turn AI chatbot built with **React**, **FastAPI**, **LiteLLM**, and **SQLite**, featuring automatic multi-key failover, response caching, per-request token tracking, and Dockerized deployment.

---

## 1. Live URLs

- **Frontend (UI):** `<PENDING_DEPLOYMENT_URL>`
- **Backend (API):** `<PENDING_DEPLOYMENT_URL>`
- **Health Check:** `<PENDING_DEPLOYMENT_URL>/health`

> Update these placeholders once you deploy the frontend (e.g., Vercel / Netlify / Cloudflare Pages) and backend (e.g., Render / Railway / Fly.io).

---

## 2. Run locally

### With Docker (recommended)

1. Copy `.env.example` to `.env` and add at least two valid API keys in `LLM_API_KEYS`:
   ```bash
   cp .env.example .env
   ```
2. Start both services (backend on `http://localhost:8000`, frontend on `http://localhost:3000`):
   ```bash
   make up
   ```
   *(On Windows without `make`, run `docker compose up -d --build`.)*
3. Stop the stack:
   ```bash
   make down
   ```
   *(Or `docker compose down`.)*

SQLite data is persisted in the named Docker volume `sqlite_data` mounted at `/data` (`DATABASE_URL=sqlite:////data/app.db`).

### Without Docker

**Backend:**
```bash
cd backend
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
export LLM_API_KEYS="your-key-1,your-key-2"
export LLM_MODEL="gemini/gemini-2.0-flash"
export DATABASE_URL="sqlite:///./app.db"
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
export ENABLE_FAILURE_SIMULATION="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev -- --port 3000
```

**Run backend tests:**
```bash
cd backend
pytest -v
```

---

## 3. Design decisions

### Key fallback
- `LLM_API_KEYS` is parsed into an ordered pool (`key_1`, `key_2`, ...), registered in LiteLLM `Router`, and attempted deterministically in order per request.
- If a key fails with HTTP `429`, a quota/resource-exhausted error, or a `5xx`/transient provider error, the gateway immediately catches the exception, increments `fallback_attempts`, and retries the next key without returning an error or stack trace to the client.
- If all keys fail, the API returns HTTP `503` with `{"detail": "all providers unavailable"}` and does **not** persist the failed turn to SQLite.
- When `ENABLE_FAILURE_SIMULATION=true`, the `X-Simulate-Failure` header injects per-key failures (`429`, `500`, `502`, `503`, or `*=<status>`) scoped strictly to that single request so no cooldown leaks across subsequent requests.

### Caching
- LiteLLM's built-in in-memory response cache (`litellm.Cache(type="local")`) is enabled on startup.
- Because the full message list sent to LiteLLM (`[system_prompt, ...history, new_user_message]`) and the target `model` determine the cache key (independent of `api_key` or `session_id`), sending the same first message in a new session returns `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, and the cached reply + token counts at sub-millisecond/low latency.
- When `X-Simulate-Failure` actively simulates a failure on a request, cache lookup is bypassed (`caching=False`) so failover and `503` behavior can be verified reliably even with repeated test prompts.

### SQLite session storage
- Uses Python's built-in `sqlite3` with WAL journal mode and a thread lock—zero external databases (no Redis, no Postgres).
- Startup creates two tables automatically:
  - `messages` (`id`, `session_id`, `role`, `content`, `metadata`, `created_at`) matching the exact minimal schema plus an index on `(session_id, id)`.
  - `sessions` (`session_id`, `system_prompt`, `created_at`, `updated_at`) so a session's optional `system_prompt` override persists across turns without leaking into `GET /sessions/{session_id}/messages`.
- A user + assistant turn is written atomically only after a successful LLM response (`422` and `503` requests never pollute history).

---

## 4. Verification

### 1. Health check
```bash
curl http://localhost:8000/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}
```

### 2. Live inference & per-message token usage
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}'
```
Expect `"resolution_source": "LIVE_INFERENCE"`, `"key_id": "key_1"`, `"fallback_attempts": 0`, and per-request `input_tokens` / `output_tokens`.

### 3. Cache hit across a new session
Send the exact same first message with a new `session_id`:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}'
```
Expect `"resolution_source": "LLM_CACHE"`, `"key_id": null`, `"fallback_attempts": 0`, identical `reply` and token counts, and significantly lower `latency_ms`.

### 4. Multi-turn memory & session history
Send a follow-up in `s-demo-1`, then inspect SQLite history:
```bash
curl http://localhost:8000/sessions/s-demo-1/messages
```
History is returned oldest-first without the system prompt.

### 5. Key failover & total outage via `X-Simulate-Failure`
```bash
# Single key failover -> 200 with key_id = "key_2", fallback_attempts = 1
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-fail-1","message":"Say hello in 3 words."}'

# All keys fail -> 503 {"detail": "all providers unavailable"} (not saved to history)
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-fail-all","message":"Say hello in 3 words."}'
```

---

## 5. Setup Used

- **Editor:** VS Code / Antigravity IDE
- **AI Tools:** Antigravity AI coding assistant for scaffolding, contract verification, and test generation

---

## 6. Bonus features (if any)

- **Failover Test Lab in the UI:** A subtle dropdown in the header bar lets reviewers trigger `key_1=429`, `key_1=500`, or `*=429` directly from the browser UI without needing `curl` (defaults to `None` for normal traffic).
- **Runtime `env.js` injection in Docker:** The Nginx frontend container injects `API_BASE_URL` at container startup while still supporting build-time `VITE_API_BASE_URL` for static hosts like Vercel/Netlify.
- **Per-key model mapping (`LLM_MODELS`):** Optional support for assigning distinct models per key (e.g., primary `gemini/gemini-2.0-flash`, backup `openai/gpt-4o-mini`) in addition to single-model multi-key pools.
