/**
 * DocuMind — Loader Component
 * Owner: Akhil (Frontend Lead)
 * Purpose: Three-dot bouncing animation shown while AI is generating a response.
 *          Animation keyframes defined in App.css (.loader-dot).
 * Connection: Used by Message.jsx when answer === "Loading…"
 */

function Loader() {
  return (
    <div style={{ display: "flex", gap: "6px" }}>
      <span className="loader-dot"></span>
      <span className="loader-dot"></span>
      <span className="loader-dot"></span>
    </div>
  );
}

export default Loader;
