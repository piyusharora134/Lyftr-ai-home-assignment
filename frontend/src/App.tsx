import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { ApiError, fetchHealth, sendChatMessage } from "./api";
import { ChatMessageItem, HealthResponse } from "./types";
import "./App.css";

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

const STARTER_PROMPTS = [
  "What is the capital of France?",
  "Explain circuit breakers in distributed systems in two sentences.",
  "Remember this code word: BLUEBIRD-42. Acknowledge briefly.",
];

export function App() {
  // Always start a fresh session on page load.
  const [sessionId, setSessionId] = useState<string>(() => createSessionId());
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((data) => {
        if (active) {
          setHealth(data);
          setHealthError(null);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setHealth(null);
          setHealthError(err instanceof Error ? err.message : "Backend unreachable");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleNewSession = () => {
    setSessionId(createSessionId());
    setMessages([]);
    setInput("");
  };

  const submitMessage = async (rawText: string) => {
    if (isLoading) return;
    const messageToSend = rawText;
    if (!messageToSend.trim()) return;

    // Clear input immediately on send as required by frontend contract.
    setInput("");

    const userMsgId = `u-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: messageToSend },
    ]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage({
        sessionId,
        message: messageToSend,
        systemPrompt,
        simulateFailure,
      });

      const assistantMsgId = `a-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          content: response.reply,
          metadata: response.metadata,
        },
      ]);
    } catch (err: unknown) {
      const statusCode = err instanceof ApiError ? err.status : undefined;
      const detail =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
          ? err.message
          : "Unexpected error contacting backend.";

      const errMsgId = `e-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      setMessages((prev) => [
        ...prev,
        {
          id: errMsgId,
          role: "error",
          content: detail,
          statusCode,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitMessage(input);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submitMessage(input);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-logo" aria-hidden="true">
            L
          </div>
          <div>
            <h1 className="brand-title">Lyftr AI Gateway</h1>
            <p className="brand-subtitle">
              Multi-turn Chatbot · LiteLLM Key Failover &amp; Cache
            </p>
          </div>
        </div>

        <div className="topbar-controls">
          <div
            className={`health-pill ${
              health ? "health-ok" : healthError ? "health-err" : "health-wait"
            }`}
            title={healthError ?? "Backend health status"}
          >
            <span className="status-dot" />
            {health ? (
              <span>
                Online · {health.keys_configured} keys ·{" "}
                {health.cache_enabled ? "Cache ON" : "Cache OFF"}
              </span>
            ) : healthError ? (
              <span>Offline</span>
            ) : (
              <span>Connecting...</span>
            )}
          </div>

          <div className="session-pill" title="Active Session ID">
            <span className="session-label">Session</span>
            <code>{sessionId}</code>
          </div>

          <button
            type="button"
            className="new-session-btn"
            onClick={handleNewSession}
            disabled={isLoading}
          >
            New Session
          </button>
        </div>
      </header>

      <section className="config-bar" aria-label="Session configuration">
        <div className="config-field config-field-grow">
          <label htmlFor="system-prompt-input">System Prompt (Optional)</label>
          <input
            id="system-prompt-input"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional system prompt for this session (e.g. You are a concise tutor)..."
          />
        </div>

        <div className="config-field">
          <label htmlFor="simulate-failure-select">Failover Hook</label>
          <select
            id="simulate-failure-select"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
          >
            <option value="">None (Normal)</option>
            <option value="key_1=429">Simulate key_1=429</option>
            <option value="key_1=500">Simulate key_1=500</option>
            <option value="*=429">Simulate *=429 (All Fail)</option>
          </select>
        </div>
      </section>

      <main className="chat-main">
        <div className="chat-stream">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a Resilient Conversation</h2>
              <p>
                Every message tracks per-request tokens, latency, cache status (
                <code>LIVE_INFERENCE</code> vs <code>LLM_CACHE</code>), and
                automatic key failover.
              </p>
              <div className="starter-grid">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="starter-chip"
                    onClick={() => void submitMessage(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => {
              if (msg.role === "user") {
                return (
                  <div key={msg.id} className="message-row message-row-user">
                    <div className="bubble bubble-user">
                      <div className="message-role">You</div>
                      <div className="message-text">{msg.content}</div>
                    </div>
                  </div>
                );
              }

              if (msg.role === "error") {
                return (
                  <div key={msg.id} className="message-row message-row-assistant">
                    <div className="bubble bubble-error" role="alert">
                      <div className="error-header">
                        Request Failed{msg.statusCode ? ` (HTTP ${msg.statusCode})` : ""}
                      </div>
                      <div className="message-text">{msg.content}</div>
                    </div>
                  </div>
                );
              }

              const meta = msg.metadata;
              const isCache = meta?.resolution_source === "LLM_CACHE";
              return (
                <div key={msg.id} className="message-row message-row-assistant">
                  <div
                    className="bubble bubble-assistant"
                    data-testid="assistant-message"
                  >
                    <div className="message-role">Assistant</div>
                    <div className="message-text">{msg.content}</div>
                    {meta && (
                      <div className="metadata-bar">
                        <span
                          className={`source-badge ${
                            isCache ? "source-cache" : "source-live"
                          }`}
                          data-testid="meta-source"
                        >
                          {meta.resolution_source}
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Model</span>
                          <span className="meta-value" data-testid="meta-model">
                            {meta.model}
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Latency</span>
                          <span className="meta-value" data-testid="meta-latency">
                            {meta.latency_ms} ms
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">In</span>
                          <span
                            className="meta-value"
                            data-testid="meta-input-tokens"
                          >
                            {meta.input_tokens}
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Out</span>
                          <span
                            className="meta-value"
                            data-testid="meta-output-tokens"
                          >
                            {meta.output_tokens}
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Key</span>
                          <span className="meta-value">
                            {meta.key_id ?? "cache"}
                          </span>
                        </span>

                        {meta.fallback_attempts > 0 && (
                          <span className="failover-badge">
                            Failover ({meta.fallback_attempts} skipped)
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {isLoading && (
            <div className="message-row message-row-assistant">
              <div className="bubble bubble-assistant bubble-loading">
                <div className="message-role">Assistant</div>
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="composer-bar">
        <form className="composer-form" onSubmit={handleSubmit}>
          <textarea
            data-testid="chat-input"
            className="chat-input"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message (Enter to send, Shift+Enter for newline)..."
            maxLength={4500}
            disabled={isLoading}
          />
          <button
            data-testid="send-button"
            className="send-button"
            type="submit"
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? "Sending..." : "Send"}
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
