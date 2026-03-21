/**
 * DocuMind — API Service Layer
 * Owner: Akhil (Frontend Lead)
 * Purpose: All HTTP calls to the FastAPI backend.
 *          Attaches JWT token from localStorage to every protected request.
 *          Handles auth endpoints, chat (streaming + non-streaming),
 *          file upload, and document management.
 * Connection: Imported by Dashboard.jsx and Login.jsx
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Auth token helpers ────────────────────────────────────────────────────

function getToken() {
  return localStorage.getItem("dm_token") || "";
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${getToken()}`,
  };
}

// Throw a structured error from a failed response
async function handleError(response) {
  let detail;
  try {
    const body = await response.json();
    detail = body?.detail || body?.error || "Request failed";
  } catch {
    detail = `HTTP ${response.status}`;
  }
  const err = new Error(
    typeof detail === "string" ? detail : detail?.error || "Request failed"
  );
  err.detail = detail;
  err.status = response.status;
  throw err;
}

// ── Auth endpoints (public — no JWT) ─────────────────────────────────────

export async function authSignup(email, password) {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) await handleError(res);
  return res.json();
}

export async function authLogin(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) await handleError(res);
  return res.json();
}

// ── Chat (non-streaming) ─────────────────────────────────────────────────

export async function askQuestion(question, chatId = null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ question, chat_id: chatId }),
  });
  if (!res.ok) await handleError(res);
  return res.json();
}

// ── Chat (streaming SSE) ─────────────────────────────────────────────────

export async function askQuestionStream(question, chatId, onToken, onMeta, onDone) {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ question, chat_id: chatId }),
  });
  if (!res.ok) await handleError(res);

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop();
    for (const line of lines) {
      const trimmed = line.replace(/^data: /, "").trim();
      if (!trimmed) continue;
      try {
        const data = JSON.parse(trimmed);
        if (data.type === "meta"  && onMeta)  onMeta(data);
        if (data.type === "token" && onToken) onToken(data.content);
        if (data.type === "done"  && onDone)  onDone();
      } catch { /* malformed SSE chunk — skip */ }
    }
  }
}

// ── File upload ───────────────────────────────────────────────────────────

export async function uploadFile(file, chatId = null) {
  const formData = new FormData();
  formData.append("file", file);

  const url = chatId
    ? `${API_BASE}/upload?chat_id=${encodeURIComponent(chatId)}`
    : `${API_BASE}/upload`;

  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },  // no Content-Type; browser sets multipart
    body: formData,
  });
  if (!res.ok) await handleError(res);
  return res.json();
}

// ── Document management ───────────────────────────────────────────────────

export async function getDocuments(chatId = null) {
  const url = chatId
    ? `${API_BASE}/documents?chat_id=${encodeURIComponent(chatId)}`
    : `${API_BASE}/documents`;
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) await handleError(res);
  return res.json();
}

export async function deleteDocument(docName, chatId = null) {
  const url = chatId
    ? `${API_BASE}/documents/${encodeURIComponent(docName)}?chat_id=${encodeURIComponent(chatId)}`
    : `${API_BASE}/documents/${encodeURIComponent(docName)}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) await handleError(res);
  return res.json();
}

// ── Health check (public) ─────────────────────────────────────────────────

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  } catch {
    return { status: "error", ollama: { available: false } };
  }
}
