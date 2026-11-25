import { ChatResponse, HealthResponse } from "./types";

declare global {
  interface Window {
    __ENV__?: {
      API_BASE_URL?: string;
    };
  }
}

export function getApiBaseUrl(): string {
  const runtimeUrl = window.__ENV__?.API_BASE_URL;
  const buildUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  const raw = (runtimeUrl || buildUrl || "http://localhost:8000").trim();
  return raw.replace(/\/+$/, "");
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function extractErrorDetail(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const loc = Array.isArray((item as { loc?: unknown }).loc)
            ? (item as { loc: unknown[] }).loc.join(".")
            : "";
          const msg = String((item as { msg: unknown }).msg);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return JSON.stringify(item);
      })
      .join("; ");
  }
  return fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBaseUrl()}/health`);
  if (!res.ok) {
    throw new ApiError(res.status, `Health check failed (${res.status})`);
  }
  return (await res.json()) as HealthResponse;
}

export async function sendChatMessage(params: {
  sessionId: string;
  message: string;
  systemPrompt?: string;
  simulateFailure?: string;
}): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (params.simulateFailure && params.simulateFailure.trim()) {
    headers["X-Simulate-Failure"] = params.simulateFailure.trim();
  }

  const body: Record<string, string> = {
    session_id: params.sessionId,
    message: params.message,
  };
  if (params.systemPrompt && params.systemPrompt.trim()) {
    body.system_prompt = params.systemPrompt.trim();
  }

  const res = await fetch(`${getApiBaseUrl()}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  const text = await res.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }

  if (!res.ok) {
    const detail = extractErrorDetail(
      parsed,
      text || `Request failed with status ${res.status}`
    );
    throw new ApiError(res.status, detail);
  }

  return parsed as ChatResponse;
}
