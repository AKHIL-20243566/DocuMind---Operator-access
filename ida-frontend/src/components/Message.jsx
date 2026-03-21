/**
 * DocuMind — Message Component
 * Owner: Akhil (Frontend Lead)
 * Purpose: Renders a single chat bubble (user or AI).
 *          AI messages support GitHub-Flavored Markdown via react-markdown.
 *          Shows animated loader while streaming is in progress.
 * Connection: Used by Dashboard.jsx in the chat message list.
 */

import Loader from "./Loader";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function Message({ role, text }) {

  const isUser = role === "user";
  const isLoading = text === "Loading…" || text === "Loading...";

  const roleClass = isUser ? "message-user" : "message-ai";

  const copyText = () => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className={roleClass}>

      <div className="message-bubble">

        {isLoading ? (
          <Loader />
        ) : (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {text}
            </ReactMarkdown>

            {!isUser && (
              <button
                className="copy-btn"
                onClick={copyText}
              >
                Copy
              </button>
            )}
          </>
        )}

      </div>

    </div>
  );
}

export default Message;