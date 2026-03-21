/**
 * DocuMind — Root App
 * Owner: Akhil (Frontend Lead)
 * Purpose: Auth routing (Login vs Dashboard) + global theme management.
 *          Theme is persisted in localStorage and applied as data-theme on <html>.
 * Connection: Renders Login.jsx or Dashboard.jsx based on JWT presence.
 */

import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import Login     from "./pages/Login";

function App() {
  // ── Auth state ────────────────────────────────────────────────────────────
  const [user, setUser] = useState(() => {
    const token = localStorage.getItem("dm_token");
    const email = localStorage.getItem("dm_email");
    return token && email ? { token, email } : null;
  });

  // ── Theme state ───────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(
    () => localStorage.getItem("dm_theme") || "dark"
  );

  // Apply theme to <html> so CSS variables take effect globally
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dm_theme", theme);
  }, [theme]);

  const toggleTheme = () =>
    setTheme((t) => (t === "dark" ? "light" : "dark"));

  // ── Auth handlers ─────────────────────────────────────────────────────────
  const handleLogin = (data) => setUser(data);

  const handleLogout = () => {
    localStorage.removeItem("dm_token");
    localStorage.removeItem("dm_email");
    setUser(null);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Dashboard
      user={user}
      theme={theme}
      onToggleTheme={toggleTheme}
      onLogout={handleLogout}
    />
  );
}

export default App;
