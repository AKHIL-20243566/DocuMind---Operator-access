import { useState, useRef, useEffect } from "react";
import ChatBox from "../components/ChatBox";
import Message from "../components/Message";
import {
  askQuestion,
  askQuestionStream,
  uploadFile,
  getDocuments,
  deleteDocument,
  checkHealth,
} from "../services/api";

import {
  PanelLeft,
  History,
  FileText,
  Upload,
  Trash2,
  FileUp,
  Zap,
  ZapOff,
  X,
  Plus,
  Menu,
  BookOpen,
  Database,
  MessageSquare,
} from "lucide-react";

function Dashboard() {

  // Sessions
  const [sessions, setSessions] = useState([
    { id: 1, title: "New Chat", messages: [] },
  ]);
  const [activeSession, setActiveSession] = useState(1);
  const [messages, setMessages] = useState([]);

  // Panel state
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 900);
  const [ragOpen, setRagOpen] = useState(window.innerWidth > 900);
  const [docPanelOpen, setDocPanelOpen] = useState(false);

  // RAG data
  const [sources, setSources] = useState([]);
  const [confidence, setConfidence] = useState(null);
  const [context, setContext] = useState([]);
  const [answerMode, setAnswerMode] = useState(null);

  // Documents
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);

  // System
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const chatAreaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Init
  useEffect(() => {
    fetchDocuments();
    checkHealth().then((h) => {
      setOllamaConnected(h?.ollama?.available || false);
    });
  }, []);

  // Session sync
  useEffect(() => {
    const session = sessions.find((s) => s.id === activeSession);
    if (session) setMessages(session.messages);
  }, [activeSession, sessions]);

  // Auto-scroll
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // Close sidebar on mobile after selection
  const closeMobileSidebar = () => {
    if (window.innerWidth <= 900) setSidebarOpen(false);
  };

  // Documents
  const fetchDocuments = async () => {
    try {
      const data = await getDocuments();
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
        const result = await uploadFile(file);
        setUploadMsg(result.message);
      }
      await fetchDocuments();
    } catch (err) {
      setUploadMsg(err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteDoc = async (docName) => {
    try {
      await deleteDocument(docName);
      await fetchDocuments();
    } catch { /* silent */ }
  };

  // New chat — reset everything including RAG panel
  const createNewChat = () => {
    const newId = Date.now();
    setSessions((prev) => [...prev, { id: newId, title: "New Chat", messages: [] }]);
    setActiveSession(newId);
    setMessages([]);
    setSources([]);
    setConfidence(null);
    setContext([]);
    setAnswerMode(null);
    closeMobileSidebar();
  };

  // Send message
  const handleSend = async (question) => {
    if (!question.trim() || loading) return;
    setLoading(true);

    const newMessage = { question, answer: "Loading..." };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSession
          ? { ...s, messages: updatedMessages, title: s.title === "New Chat" ? question.slice(0, 30) : s.title }
          : s
      )
    );

    try {
      if (ollamaConnected) {
        let fullAnswer = "";
        await askQuestionStream(
          question,
          (token) => {
            fullAnswer += token;
            const updated = [...updatedMessages];
            updated[updated.length - 1] = { question, answer: fullAnswer };
            setMessages([...updated]);
          },
          (meta) => {
            setSources(meta.sources || []);
            setConfidence(meta.confidence ?? null);
            setContext(meta.context || []);
            setAnswerMode(meta.mode || null);
          },
          () => {
            const final_ = [...updatedMessages];
            final_[final_.length - 1] = { question, answer: fullAnswer || "No response." };
            setMessages([...final_]);
            setSessions((prev) =>
              prev.map((s) => (s.id === activeSession ? { ...s, messages: final_ } : s))
            );
          }
        );
      } else {
        const response = await askQuestion(question);
        const updated = [...updatedMessages];
        updated[updated.length - 1].answer = response?.answer || "No response from AI.";
        setMessages(updated);
        setSessions((prev) =>
          prev.map((s) => (s.id === activeSession ? { ...s, messages: updated } : s))
        );
        setSources(response?.sources || []);
        setConfidence(response?.confidence ?? null);
        setContext(response?.context || []);
        setAnswerMode(response?.mode || null);
      }
    } catch {
      const updated = [...updatedMessages];
      updated[updated.length - 1].answer = "Error contacting AI server.";
      setMessages(updated);
    } finally {
      setLoading(false);
    }
  };

  // Logo click → full page reload
  const handleLogoClick = () => {
    window.location.reload();
  };

  return (
    <div className="app-layout">

      {/* Mobile overlay */}
      {sidebarOpen && window.innerWidth <= 900 && (
        <div className="mobile-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Left rail */}
      <div className="left-rail">
        <button className="rail-btn" onClick={() => setSidebarOpen(!sidebarOpen)} title="Chat History">
          {window.innerWidth <= 900 ? <Menu size={22} strokeWidth={2} /> : <PanelLeft size={22} strokeWidth={2} />}
        </button>
        <button className="rail-btn" onClick={() => setDocPanelOpen(!docPanelOpen)} title="Documents">
          <FileUp size={22} strokeWidth={2} />
        </button>
      </div>

      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand" onClick={handleLogoClick} style={{ cursor: "pointer" }}>
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
              onClick={() => { setActiveSession(session.id); closeMobileSidebar(); }}
            >
              <MessageSquare size={13} />
              <span>{session.title}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Document panel */}
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
              accept=".pdf,.docx,.txt,.csv,.md"
              multiple
              hidden
            />
            <button
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              <Upload size={16} />
              {uploading ? "Uploading..." : "Upload Documents"}
            </button>
            <p className="upload-hint">PDF, DOCX, TXT, CSV, Markdown — max 10 MB</p>
            {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}
          </div>

          <div className="doc-list">
            {documents.length === 0 ? (
              <p className="doc-empty">No documents uploaded yet.</p>
            ) : (
              documents.map((doc, i) => (
                <div key={i} className="doc-item">
                  <div className="doc-info">
                    <span className="doc-name"><FileText size={13} /> {doc.name}</span>
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

      {/* Chat */}
      <div className="chat-container">
        <div className="chat-header">
          <h1 onClick={handleLogoClick} style={{ cursor: "pointer" }}>DocuMind</h1>
          <div className="status-badges">
            <span className={`status-badge ${ollamaConnected ? "connected" : "disconnected"}`}>
              {ollamaConnected ? <Zap size={12} /> : <ZapOff size={12} />}
              {ollamaConnected ? "Ollama" : "Fallback"}
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
              <p>Upload documents and ask questions. The AI searches your documents and generates grounded answers.</p>
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
          {messages.map((msg, index) => (
            <div key={index}>
              <Message role="user" text={msg.question} />
              <Message role="ai" text={msg.answer} />
            </div>
          ))}
        </div>

        <ChatBox onSend={handleSend} loading={loading} />
      </div>

      {/* RAG panel */}
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
                  <div className="source-title"><FileText size={13} /> {src.doc}</div>
                  <div className="source-meta">Page {src.page} — Score: {(src.score * 100).toFixed(0)}%</div>
                  <div className="source-bar">
                    <div className="source-fill" style={{ width: `${(src.score || 0) * 100}%` }} />
                  </div>
                </div>
              ))
            )}
          </div>

          <h4>Confidence</h4>
          <div className="confidence-wrapper">
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: confidence ? `${confidence * 100}%` : "0%" }} />
            </div>
            {confidence !== null && <span className="confidence-label">{(confidence * 100).toFixed(0)}%</span>}
          </div>

          <h4>Context Preview</h4>
          <div className="context-preview">
            {context.length === 0
              ? "No retrieved context yet."
              : context.map((c, i) => (
                  <div key={i} className="context-chunk">
                    <p>{c}</p>
                  </div>
                ))
            }
          </div>
        </div>
      )}

      {/* Right rail */}
      <div className="right-rail">
        <button className="rail-btn" onClick={() => setRagOpen(!ragOpen)} title="RAG Panel">
          <FileText size={22} strokeWidth={2} />
        </button>
      </div>
    </div>
  );
}

export default Dashboard;