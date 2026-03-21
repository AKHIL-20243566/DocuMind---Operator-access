/**
 * DocuMind — Login / Signup Page
 * Owner: Akhil (Frontend Lead)
 * Purpose: Authenticate users with @mnnit.ac.in email restriction.
 *          Stores JWT token in localStorage on success.
 *          Includes interactive particle canvas background + custom cursor.
 * Connection: Rendered by App.jsx when no valid token is found.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { authSignup, authLogin } from "../services/api";

// ── Particle system constants ──────────────────────────────────────────────
const PARTICLE_COUNT  = 90;
const MAX_LINK_DIST   = 140;
const MOUSE_REPEL_R   = 120;
const MOUSE_ATTRACT_R = 220;
const BASE_SPEED      = 0.35;

function initParticles(w, h) {
  return Array.from({ length: PARTICLE_COUNT }, () => ({
    x:  Math.random() * w,
    y:  Math.random() * h,
    vx: (Math.random() - 0.5) * BASE_SPEED * 2,
    vy: (Math.random() - 0.5) * BASE_SPEED * 2,
    r:  Math.random() * 2 + 1,
    opacity: Math.random() * 0.5 + 0.3,
  }));
}

export default function Login({ onLogin }) {
  const [mode, setMode]         = useState("login");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);

  const canvasRef   = useRef(null);
  const mouseRef    = useRef({ x: -1000, y: -1000 });
  const particleRef = useRef([]);
  const rafRef      = useRef(null);

  // ── Canvas particle loop ───────────────────────────────────────────────
  const startCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      particleRef.current = initParticles(canvas.width, canvas.height);
    };
    resize();
    window.addEventListener("resize", resize);

    const accentRGB = "99,102,241"; // indigo-500

    const tick = () => {
      const W = canvas.width;
      const H = canvas.height;
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      ctx.clearRect(0, 0, W, H);

      const ps = particleRef.current;

      // Update positions
      for (const p of ps) {
        // Mouse interaction — repel close, gently attract far
        const dx = p.x - mx;
        const dy = p.y - my;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        if (dist < MOUSE_REPEL_R) {
          const force = (MOUSE_REPEL_R - dist) / MOUSE_REPEL_R * 0.06;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        } else if (dist < MOUSE_ATTRACT_R) {
          const force = (dist - MOUSE_REPEL_R) / MOUSE_ATTRACT_R * 0.004;
          p.vx -= (dx / dist) * force;
          p.vy -= (dy / dist) * force;
        }

        // Speed cap
        const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (speed > BASE_SPEED * 4) {
          p.vx = (p.vx / speed) * BASE_SPEED * 4;
          p.vy = (p.vy / speed) * BASE_SPEED * 4;
        }

        // Dampen back to base speed
        p.vx *= 0.99;
        p.vy *= 0.99;

        p.x += p.vx;
        p.y += p.vy;

        // Wrap edges
        if (p.x < 0) p.x = W;
        if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H;
        if (p.y > H) p.y = 0;
      }

      // Draw connection lines
      for (let i = 0; i < ps.length; i++) {
        for (let j = i + 1; j < ps.length; j++) {
          const dx   = ps[i].x - ps[j].x;
          const dy   = ps[i].y - ps[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MAX_LINK_DIST) {
            const alpha = (1 - dist / MAX_LINK_DIST) * 0.25;
            ctx.strokeStyle = `rgba(${accentRGB},${alpha})`;
            ctx.lineWidth   = 0.8;
            ctx.beginPath();
            ctx.moveTo(ps[i].x, ps[i].y);
            ctx.lineTo(ps[j].x, ps[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw particles
      for (const p of ps) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${accentRGB},${p.opacity})`;
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    const cleanup = startCanvas();
    return cleanup;
  }, [startCanvas]);

  const handleMouseMove = (e) => {
    mouseRef.current = { x: e.clientX, y: e.clientY };
    // Move cursor dot
    const dot = document.getElementById("cursor-dot");
    if (dot) {
      dot.style.left = e.clientX + "px";
      dot.style.top  = e.clientY + "px";
    }
  };

  // ── Form logic ─────────────────────────────────────────────────────────
  const isValidEmail = (e) => e.trim().toLowerCase().endsWith("@mnnit.ac.in");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!isValidEmail(email)) {
      setError("Only @mnnit.ac.in email addresses are allowed.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setLoading(true);
    try {
      const fn   = mode === "signup" ? authSignup : authLogin;
      const data = await fn(email.trim().toLowerCase(), password);
      localStorage.setItem("dm_token", data.token);
      localStorage.setItem("dm_email", data.email);
      onLogin(data);
    } catch (err) {
      setError(err.message || "Authentication failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page" onMouseMove={handleMouseMove}>
      {/* Interactive canvas background */}
      <canvas ref={canvasRef} className="auth-canvas" />

      {/* Cursor dot */}
      <div className="cursor-dot" id="cursor-dot" />

      <div className="auth-card">
        {/* Brand */}
        <div className="auth-brand">
          <h1>DocuMind</h1>
          <p>Internal Knowledge Assistant — MNNIT</p>
        </div>

        {/* Tab switcher */}
        <div className="auth-tabs">
          <button
            className={mode === "login" ? "auth-tab active" : "auth-tab"}
            onClick={() => { setMode("login"); setError(""); }}
          >
            Login
          </button>
          <button
            className={mode === "signup" ? "auth-tab active" : "auth-tab"}
            onClick={() => { setMode("signup"); setError(""); }}
          >
            Sign Up
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-field">
            <label>Institutional Email</label>
            <input
              type="email"
              placeholder="yourname@mnnit.ac.in"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>

          <div className="auth-field">
            <label>Password</label>
            <input
              type="password"
              placeholder={mode === "signup" ? "Minimum 8 characters" : "Enter password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
            />
          </div>

          {error && <p className="auth-error">{error}</p>}

          <button type="submit" className="auth-submit" disabled={loading}>
            {loading
              ? (mode === "signup" ? "Creating account…" : "Signing in…")
              : (mode === "signup" ? "Create Account" : "Sign In")}
          </button>
        </form>

        <p className="auth-footer">
          Access restricted to{" "}
          <span className="auth-domain">@mnnit.ac.in</span> accounts
        </p>
      </div>
    </div>
  );
}
