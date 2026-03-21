/**
 * DocuMind — Main Dashboard
 * Owner: Akhil (Frontend Lead + Partial RAG Integration)
 * Purpose: Full chat UI with session management, document upload, RAG insights panel,
 *          theme toggle, settings panel (security dashboard + logout), and
 *          per-chat document isolation via chatId.
 * Connection: Rendered by App.jsx after successful auth.
 *             Calls api.js for all backend operations.
 */

import { useState, useRef, useEffect } from "react";
import ChatBox  from "../components/ChatBox";
import Message  from "../components/Message";
import {
  askQuestion, askQuestionStream,
  uploadFile, getDocuments, deleteDocument, checkHealth,
} from "../services/api";
import {
  PanelLeft, History, FileText, Upload, Trash2, FileUp,
  Zap, ZapOff, X, Plus, Menu, BookOpen, Database,
  MessageSquare, Settings, Sun, Moon, Shield, LogOut,
} from "lucide-react";

// ── Query log stored in localStorage for the security dashboard ───────────
const QUERY_LOG_KEY = "dm_query_log";
const MAX_LOG_ENTRIES = 50;

function appendQueryLog(entry) {
  try {
    const existing = JSON.parse(localStorage.getItem(QUERY_LOG_KEY) || "[]");
    const updated  = [entry, ...existing].slice(0, MAX_LOG_ENTRIES);
    localStorage.setItem(QUERY_LOG_KEY, JSON.stringify(updated));
  } catch { /* silent */ }
}

function getQueryLog() {
  try {
    return JSON.parse(localStorage.getItem(QUERY_LOG_KEY) || "[]");
  } catch { return []; }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function maskEmail(email = "") {
  const [local, domain] = email.split("@");
  if (!local || !domain) return email;
  return `${local[0]}***@${domain}`;
}

function formatTimestamp(iso) {
  return new Date(iso).toLocaleString();
}

// ── Dashboard ─────────────────────────────────────────────────────────────

export default function Dashboard({ user, theme, onToggleTheme, onLogout }) {
  // Sessions — each session is a chat with its own chatId and message list
  const [sessions, setSessions]         = useState([
    { id: String(Date.now()), title: "New Chat", messages: [] },
  ]);
  const [activeSession, setActiveSession] = useState(sessions[0].id);
  const [messages, setMessages]           = useState([]);

  // Panel visibility
  const [sidebarOpen,  setSidebarOpen]  = useState(window.innerWidth > 900);
  const [ragOpen,      setRagOpen]      = useState(window.innerWidth > 900);
  const [docPanelOpen, setDocPanelOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // RAG state
  const [sources,     setSources]     = useState([]);
  const [confidence,  setConfidence]  = useState(null);
  const [context,     setContext]     = useState([]);
  const [answerMode,  setAnswerMode]  = useState(null);

  // Documents
  const [documents,  setDocuments]  = useState([]);
  const [uploading,  setUploading]  = useState(false);
  const [uploadMsg,  setUploadMsg]  = useState(null);

  // System
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [loading,         setLoading]         = useState(false);

  // Security dashboard
  const [queryLog, setQueryLog] = useState([]);

  const chatAreaRef  = useRef(null);
  const fileInputRef = useRef(null);

  // ── Active session helpers ─────────────────────────────────────────────

  const activeSessionData = sessions.find((s) => s.id === activeSession);
  const chatId = activeSession;   // session id == chatId for backend isolation

  // ── Init ──────────────────────────────────────────────────────────────

  useEffect(() => {
    fetchDocuments();

    // Check Ollama on mount and every 15 seconds so badge auto-updates
    // when user starts Ollama after the page loads
    const pollHealth = () =>
      checkHealth().then((h) => setOllamaConnected(h?.ollama?.available || false));

    pollHealth();
    const healthInterval = setInterval(pollHealth, 15000);
    return () => clearInterval(healthInterval);
  }, []);

  // Sync messages when switching sessions
  useEffect(() => {
    const s = sessions.find((s) => s.id === activeSession);
    if (s) setMessages(s.messages);
    fetchDocuments();
    setSources([]); setConfidence(null); setContext([]); setAnswerMode(null);
  }, [activeSession]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatAreaRef.current)
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
  }, [messages]);

  // ── Documents ─────────────────────────────────────────────────────────

  const fetchDocuments = async () => {
    try {
      const data = await getDocuments(chatId);
      setDocuments(data.documents || []);
    } catch { /* silent */ }
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadMsg(null);
    try {
      for (const file of files) {
        const result = await uploadFile(file, chatId);
        const msgs   = Array.isArray(result.status_messages) ? result.status_messages : [];
        const ocrTag = result.ocr_triggered ? " [OCR]" : "";
        setUploadMsg(msgs.length ? `${msgs.join(" → ")}${ocrTag} | ${result.message}` : result.message);
      }
      await fetchDocuments();
    } catch (err) {
      const detail = err?.detail;
      if (detail && typeof detail === "object") {
        const msgs = Array.isArray(detail.status_messages) ? detail.status_messages : [];
        setUploadMsg(msgs.length ? `${msgs.join(" → ")} | ${detail.error || ""}` : detail.error || "Upload failed");
      } else {
        setUploadMsg(err.message || "Upload failed");
      }
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteDoc = async (docName) => {
    // Optimistic update — remove from UI instantly, no waiting for backend
    setDocuments((prev) => prev.filter((d) => d.name !== docName));
    try {
      await deleteDocument(docName, chatId);
    } catch {
      // Rollback only if the delete actually failed
      fetchDocuments();
    }
  };

  // ── Session management ────────────────────────────────────────────────

  const createNewChat = () => {
    const newId = String(Date.now());
    setSessions((prev) => [...prev, { id: newId, title: "New Chat", messages: [] }]);
    setActiveSession(newId);
    setMessages([]);
    if (window.innerWidth <= 900) setSidebarOpen(false);
  };

  const updateSession = (id, updatedMessages, question) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === id
          ? {
              ...s,
              messages: updatedMessages,
              title: s.title === "New Chat" && question
                ? question.slice(0, 32)
                : s.title,
            }
          : s
      )
    );
  };

  // ── Send message ──────────────────────────────────────────────────────

  const handleSend = async (question) => {
    if (!question.trim() || loading) return;
    setLoading(true);

    const newMessage     = { question, answer: "Loading…" };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    updateSession(activeSession, updatedMessages, question);

    const logEntry = {
      query: question,
      timestamp: new Date().toISOString(),
      chat_id: chatId,
    };

    try {
      if (ollamaConnected) {
        let fullAnswer = "";

        await askQuestionStream(
          question,
          chatId,
          (token) => {
            fullAnswer += token;
            const u = [...updatedMessages];
            u[u.length - 1] = { question, answer: fullAnswer };
            setMessages([...u]);
          },
          (meta) => {
            setSources(meta.sources || []);
            setConfidence(meta.confidence ?? null);
            setContext(meta.context  || []);
            setAnswerMode(meta.mode  || null);
            // Enrich log with retrieval metadata
            logEntry.mode    = meta.mode;
            logEntry.sources = (meta.sources || []).map((s) => s.doc);
          },
          () => {
            const fin = [...updatedMessages];
            fin[fin.length - 1] = { question, answer: fullAnswer || "No response." };
            setMessages([...fin]);
            updateSession(activeSession, fin);
          }
        );
      } else {
        const response = await askQuestion(question, chatId);
        const fin      = [...updatedMessages];
        fin[fin.length - 1].answer = response?.answer || "No response from AI.";
        setMessages(fin);
        updateSession(activeSession, fin);
        setSources(response?.sources   || []);
        setConfidence(response?.confidence ?? null);
        setContext(response?.context   || []);
        setAnswerMode(response?.mode   || null);
        logEntry.mode    = response?.mode;
        logEntry.sources = (response?.sources || []).map((s) => s.doc);
      }
    } catch (err) {
      const fin = [...updatedMessages];
      // Handle 401 gracefully
      if (err.status === 401) {
        fin[fin.length - 1].answer = "Session expired. Please log in again.";
        setTimeout(onLogout, 1500);
      } else {
        fin[fin.length - 1].answer = "Error contacting AI server. Please try again.";
      }
      setMessages(fin);
    } finally {
      setLoading(false);
      appendQueryLog(logEntry);
    }
  };

  // ── Settings panel ────────────────────────────────────────────────────

  const openSettings = () => {
    setQueryLog(getQueryLog());
    setSettingsOpen(true);
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="app-layout">

      {/* Mobile overlay for sidebar */}
      {sidebarOpen && window.innerWidth <= 900 && (
        <div className="mobile-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Settings modal overlay */}
      {settingsOpen && (
        <div className="settings-overlay" onClick={() => setSettingsOpen(false)} />
      )}

      {/* ── Left rail ─────────────────────────────────────────────── */}
      <div className="left-rail">
        <button
          className="rail-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          title="Chat History"
        >
          {window.innerWidth <= 900
            ? <Menu size={22} strokeWidth={2} />
            : <PanelLeft size={22} strokeWidth={2} />}
        </button>
        <button
          className="rail-btn"
          onClick={() => setDocPanelOpen(!docPanelOpen)}
          title="Documents"
        >
          <FileUp size={22} strokeWidth={2} />
        </button>
        {/* Theme toggle */}
        <button
          className="rail-btn"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {theme === "dark"
            ? <Sun  size={22} strokeWidth={2} />
            : <Moon size={22} strokeWidth={2} />}
        </button>
        {/* Settings */}
        <button className="rail-btn" onClick={openSettings} title="Settings">
          <Settings size={22} strokeWidth={2} />
        </button>
      </div>

      {/* ── Sidebar ───────────────────────────────────────────────── */}
      <div className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <h2>DocuMind</h2>
          <p>Knowledge Assistant</p>
        </div>
        <button className="new-chat-btn" onClick={createNewChat}>
          <Plus size={14} /> New Chat
        </button>
        <p className="sidebar-title"><History size={14} /> Chat History</p>
        <ul className="sidebar-list">
          {sessions.map((session) => (
            <li
              key={session.id}
              className={session.id === activeSession ? "active-session" : ""}
              onClick={() => {
                setActiveSession(session.id);
                if (window.innerWidth <= 900) setSidebarOpen(false);
              }}
            >
              <MessageSquare size={13} />
              <span>{session.title}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* ── Document panel ────────────────────────────────────────── */}
      {docPanelOpen && (
        <div className="doc-panel">
          <div className="doc-panel-header">
            <h3>Documents</h3>
            <button className="icon-btn" onClick={() => setDocPanelOpen(false)}>
              <X size={18} />
            </button>
          </div>
          <div className="upload-area">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleUpload}
              accept=".pdf,.docx,.txt,.csv,.md,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
              multiple
              hidden
            />
            <button
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              <Upload size={16} />
              {uploading ? "Uploading…" : "Upload Documents"}
            </button>
            <p className="upload-hint">PDF · DOCX · TXT · CSV · MD · PNG · JPG · TIFF — max 10 MB</p>
            <p className="upload-hint" style={{ fontSize: "0.7rem", opacity: 0.6 }}>Scanned PDFs &amp; images are processed via OCR</p>
            {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}
          </div>
          <div className="doc-list">
            {documents.length === 0 ? (
              <p className="doc-empty">No documents uploaded yet.</p>
            ) : (
              documents.map((doc, i) => (
                <div key={i} className="doc-item">
                  <div className="doc-info">
                    <span className="doc-name">
                      <FileText size={13} /> {doc.name}
                    </span>
                    <span className="doc-meta">{doc.chunks} chunks</span>
                  </div>
                  <button className="doc-delete" onClick={() => handleDeleteDoc(doc.name)}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* ── Chat area ─────────────────────────────────────────────── */}
      <div className="chat-container">
        <div className="chat-header">
          <h1>DocuMind</h1>
          <div className="status-badges">
            <span
              className={`status-badge ${ollamaConnected ? "connected" : "disconnected"}`}
              onClick={() => checkHealth().then((h) => setOllamaConnected(h?.ollama?.available || false))}
              title="Click to retry Ollama connection"
              style={{ cursor: "pointer" }}
            >
              {ollamaConnected ? <Zap size={12} /> : <ZapOff size={12} />}
              {ollamaConnected ? "Ollama" : "Fallback — click to retry"}
            </span>
            {answerMode && (
              <span className={`mode-badge mode-${answerMode}`}>
                {answerMode === "rag" ? "RAG" : "LLM"}
              </span>
            )}
          </div>
        </div>

        <div className="chat-area" ref={chatAreaRef}>
          {messages.length === 0 && (
            <div className="empty-chat">
              <h2>Welcome to DocuMind</h2>
              <p>Upload documents and ask questions. Answers are grounded strictly in your documents.</p>
              <div className="suggestions">
                <button onClick={() => handleSend("Summarize the uploaded document")}>
                  <BookOpen size={14} /> Summarize document
                </button>
                <button onClick={() => handleSend("What are the key ideas in this document?")}>
                  <FileText size={14} /> Key ideas
                </button>
                <button onClick={() => handleSend("Explain the main topics covered")}>
                  <Database size={14} /> Main topics
                </button>
              </div>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx}>
              <Message role="user" text={msg.question} />
              <Message role="ai"   text={msg.answer}   />
            </div>
          ))}
        </div>

        <ChatBox onSend={handleSend} loading={loading} />
      </div>

      {/* ── RAG insights panel ────────────────────────────────────── */}
      {ragOpen && (
        <div className="rag-panel">
          <h3>Retrieval Insights</h3>

          <h4>Sources</h4>
          <div className="source-list">
            {sources.length === 0 ? (
              <p className="source-empty">No sources yet</p>
            ) : (
              sources.map((src, i) => (
                <div key={i} className="source-card">
                  <div className="source-title">
                    <FileText size={13} />
                    <span title={src.doc}>{src.doc}</span>
                  </div>
                  <div className="source-meta">
                    Page {src.page} — Score: {(src.score * 100).toFixed(0)}%
                  </div>
                  <div className="source-bar">
                    <div
                      className="source-fill"
                      style={{ width: `${Math.min((src.score || 0) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>

          <h4>Confidence</h4>
          <div className="confidence-wrapper">
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: confidence !== null ? `${confidence * 100}%` : "0%" }}
              />
            </div>
            {confidence !== null && (
              <span className="confidence-label">{(confidence * 100).toFixed(0)}%</span>
            )}
          </div>

          <h4>Context Preview</h4>
          <div className="context-preview">
            {context.length === 0
              ? "No retrieved context yet."
              : context.map((c, i) => (
                  <div key={i} className="context-chunk">
                    <p>{c}</p>
                  </div>
                ))}
          </div>
        </div>
      )}

      {/* ── Right rail ────────────────────────────────────────────── */}
      <div className="right-rail">
        <button
          className="rail-btn"
          onClick={() => setRagOpen(!ragOpen)}
          title="RAG Panel"
        >
          <FileText size={22} strokeWidth={2} />
        </button>
      </div>

      {/* ── Settings panel ────────────────────────────────────────── */}
      {settingsOpen && (
        <div className="settings-panel">
          <div className="settings-header">
            <h3><Shield size={15} /> Settings &amp; Security</h3>
            <button className="icon-btn" onClick={() => setSettingsOpen(false)}>
              <X size={18} />
            </button>
          </div>

          {/* Account */}
          <div className="settings-section">
            <h4>Account</h4>
            <div className="settings-row">
              <span className="settings-label">Logged in as</span>
              <span className="settings-value">{maskEmail(user?.email)}</span>
            </div>
            <div className="settings-row">
              <span className="settings-label">Theme</span>
              <button className="theme-toggle-btn" onClick={onToggleTheme}>
                {theme === "dark"
                  ? <><Sun size={13} /> Light Mode</>
                  : <><Moon size={13} /> Dark Mode</>}
              </button>
            </div>
          </div>

          {/* Security dashboard */}
          <div className="settings-section">
            <h4><Shield size={12} /> Security Dashboard</h4>
            <div className="security-info">
              <div className="settings-row">
                <span className="settings-label">Email (masked)</span>
                <span className="settings-value mono">{maskEmail(user?.email)}</span>
              </div>
              <div className="settings-row">
                <span className="settings-label">Rate limit</span>
                <span className="settings-value">5 req / min</span>
              </div>
              <div className="settings-row">
                <span className="settings-label">Auth</span>
                <span className="settings-value">JWT · @mnnit.ac.in only</span>
              </div>
            </div>

            <h4 style={{ marginTop: "12px" }}>Recent Queries</h4>
            <div className="query-log">
              {queryLog.length === 0 ? (
                <p className="log-empty">No queries logged yet.</p>
              ) : (
                queryLog.slice(0, 10).map((entry, i) => (
                  <div key={i} className="log-entry">
                    <div className="log-query">"{entry.query}"</div>
                    <div className="log-meta">
                      {formatTimestamp(entry.timestamp)}
                      {entry.mode && <span className={`log-mode mode-${entry.mode}`}>{entry.mode.toUpperCase()}</span>}
                    </div>
                    {entry.sources?.length > 0 && (
                      <div className="log-sources">
                        {entry.sources.map((s, j) => <span key={j} className="log-source-tag">{s}</span>)}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Logout */}
          <button className="logout-btn" onClick={onLogout}>
            <LogOut size={14} /> Sign Out
          </button>
        </div>
      )}

    </div>
  );
}
