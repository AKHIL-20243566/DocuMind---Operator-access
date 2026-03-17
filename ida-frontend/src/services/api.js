const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function askQuestion(question) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  if (!response.ok) throw new Error("Server error");
  return await response.json();
}

export async function askQuestionStream(question, onToken, onMeta, onDone) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  if (!response.ok) throw new Error("Server error");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

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
        if (data.type === "meta" && onMeta) onMeta(data);
        else if (data.type === "token" && onToken) onToken(data.content);
        else if (data.type === "done" && onDone) onDone();
      } catch {}
    }
  }
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Upload failed");
  }
  return await response.json();
}

export async function getDocuments() {
  const response = await fetch(`${API_BASE}/documents`);
  if (!response.ok) throw new Error("Failed to fetch documents");
  return await response.json();
}

export async function deleteDocument(docName) {
  const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(docName)}`, {
    method: "DELETE"
  });
  if (!response.ok) throw new Error("Failed to delete document");
  return await response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return await response.json();
  } catch {
    return { status: "error", ollama: { available: false } };
  }
}