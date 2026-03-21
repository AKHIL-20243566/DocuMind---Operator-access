/**
 * DocuMind — ChatBox Component
 * Owner: Akhil (Frontend Lead)
 * Purpose: Text input + send button at the bottom of the chat area.
 *          Submits on Enter or click; auto-focuses after each send.
 * Connection: Used by Dashboard.jsx; calls onSend(question) prop.
 */

import { useState, useRef, useEffect } from "react";
import { ArrowRight } from "lucide-react";
function ChatBox({ onSend, loading }) {

  const [question, setQuestion] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = () => {

    if (!question.trim() || loading) return;

    onSend(question);
    setQuestion("");

    // Auto focus after sending
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <div className="chatbox">

      <input
        ref={inputRef}
        type="text"
        placeholder="Ask a question about your documents..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
      />

    <button
      onClick={handleSubmit}
      disabled={loading}
      className="send-btn"
    >
      <ArrowRight size={20} color="#ffffff" strokeWidth={2} />
    </button>

    </div>
  );
}

export default ChatBox;