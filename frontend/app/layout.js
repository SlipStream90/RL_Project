"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import "./globals.css";
import { pauseAnimations, resumeAnimations, onResume } from "./motion";

const NAV = [
  ["/marketplace", "Marketplace"],
  ["/agent-profiles", "Agent Profiles"],
  ["/negotiation-replay", "Negotiation Replay"],
  ["/explainability", "Explainability"],
  ["/experiment-lab", "Experiment Lab"],
  ["/how-it-works", "How it works"],
];

const AGENTS = [
  { id: "buyer_a", name: "Buyer A", role: "buyer", desc: "Price-focused buyer — wants resources cheaply, stays below reservation price ₹85." },
  { id: "buyer_b", name: "Buyer B", role: "buyer", desc: "Speed-focused buyer — wants resources quickly, accepts faster to close deals." },
  { id: "seller_a", name: "Seller A", role: "seller", desc: "Margin-focused seller — maximizes profit per unit, won't sell below unit cost ₹60." },
  { id: "seller_b", name: "Seller B", role: "seller", desc: "Volume-focused seller — maximizes units sold, accepts lower prices to move inventory." },
  { id: "market_manager", name: "Market Manager", role: "manager", desc: "Market stability — keeps price variance low and transaction rate high." },
];

function roleChipClass(role) {
  if (role === "buyer") return "buyer";
  if (role === "seller") return "seller";
  return "manager";
}

// ------------------------------------------------------------
// Theme toggle (persists in localStorage, try/catch)
// ------------------------------------------------------------
function ThemeToggle() {
  // Deterministic initial value (server and client agree); stored choice loads after mount.
  const [theme, setTheme] = useState("system");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("nexus_theme");
      if (saved === "light" || saved === "dark" || saved === "system") setTheme(saved);
    } catch { /* storage unavailable */ }
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const effective = theme === "system" ? (systemDark ? "dark" : "light") : theme;
    root.setAttribute("data-theme", effective);
    root.style.colorScheme = effective;
  }, [theme]);

  const cycle = () => {
    const next = theme === "system" ? "light" : theme === "light" ? "dark" : "system";
    try { localStorage.setItem("nexus_theme", next); } catch { /* storage unavailable */ }
    setTheme(next);
  };

  const labels = { system: "◐ Auto", light: "☀ Light", dark: "☾ Dark" };
  return (
    <button
      className="theme-toggle"
      onClick={cycle}
      aria-label={`Theme: ${theme}. Activate to switch theme.`}
      title={`Theme: ${theme} — click to switch (Auto → Light → Dark)`}
      data-testid="theme-toggle"
    >
      {labels[theme]}
    </button>
  );
}

// ------------------------------------------------------------
// How to use NEXUS checklist (ticked from real state)
// ------------------------------------------------------------
function GuideChecklist() {
  const [done, setDone] = useState({
    started: false,
    stepped: false,
    replayed: false,
    queried: false,
  });
  const [collapsed, setCollapsed] = useState(false);

  const refresh = () => {
    try {
      setDone({
        started: !!localStorage.getItem("nexus_episode"),
        stepped: Number(localStorage.getItem("nexus_turns") || "0") > 0,
        replayed: localStorage.getItem("nexus_replayed") === "1",
        queried: localStorage.getItem("nexus_queried") === "1",
      });
    } catch { /* storage unavailable */ }
  };

  useEffect(() => { refresh(); }, []);

  const steps = [
    { id: "started", label: "Start an episode" },
    { id: "stepped", label: "Step through turns" },
    { id: "replayed", label: "Read the replay" },
    { id: "queried", label: "Ask “why?” in Explainability" },
  ];

  const computed = steps.map((s) => !!done[s.id]);
  const allDone = computed.every(Boolean);
  const next = steps.find((_, i) => !computed[i]);

  // Listen for episode changes dispatched from marketplace page
  useEffect(() => {
    window.addEventListener("nexus-episode-changed", refresh);
    window.addEventListener("focus", refresh);
    return () => {
      window.removeEventListener("nexus-episode-changed", refresh);
      window.removeEventListener("focus", refresh);
    };
  }, []);

  return (
    <section className="panel guide-checklist" style={{ position: "sticky", top: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <h3 style={{ margin: 0, fontSize: "0.95rem" }}>How to use NEXUS</h3>
        <button
          className="ghost"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand guide" : "Collapse guide"}
          style={{ padding: "0.2rem 0.5rem", fontSize: "0.7rem" }}
        >
          {collapsed ? "▾" : "▴"}
        </button>
      </div>
      {!collapsed && (
        <>
          <ol style={{ listStyle: "none", padding: 0, margin: "0 0 0.75rem 0", fontSize: "0.82rem", color: "var(--muted)", lineHeight: 1.6 }}>
            {steps.map((s, i) => (
              <li key={s.id} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.4rem" }}>
                <span
                  data-anim="checklist-tick"
                  style={{
                    fontWeight: 700,
                    color: computed[i] ? "var(--role-buyer)" : "var(--ink)",
                    width: "20px",
                    textAlign: "center",
                  }}
                >
                  {computed[i] ? "✓" : "○"}
                </span>
                <span style={{ textDecoration: computed[i] ? "line-through" : "none", color: computed[i] ? "var(--muted)" : "inherit" }}>
                  {s.label}
                </span>
              </li>
            ))}
          </ol>
          {next && (
            <p style={{ fontSize: "0.75rem", color: "var(--muted)", margin: 0 }}>
              Next action: <strong style={{ color: "var(--ink)" }}>{next.label}</strong>
            </p>
          )}
          {allDone && (
            <p style={{ fontSize: "0.75rem", color: "var(--role-buyer)", margin: 0, fontWeight: 600 }}>
              All steps complete — you’ve seen the full loop.
            </p>
          )}
        </>
      )}
    </section>
  );
}

// ------------------------------------------------------------
// Welcome panel (first-visit, localStorage try/catch)
// ------------------------------------------------------------
function WelcomePanel() {
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    try { setDismissed(localStorage.getItem("nexus_welcome_dismissed") === "1"); } catch { }
  }, []);

  if (dismissed) return null;

  return (
    <section className="panel" aria-label="Welcome guide">
      <h3 style={{ marginTop: 0, fontSize: "0.95rem" }}>Welcome to NEXUS</h3>
      <p style={{ margin: "0.4rem 0 0.5rem", fontSize: "0.85rem", color: "var(--muted)" }}>
        NEXUS simulates a five-agent marketplace where RL policies negotiate. Every step is recorded, so you can replay it and ask why an agent acted.
      </p>
      <ul style={{ margin: "0 0 0.75rem 1.2rem", padding: 0, fontSize: "0.82rem", color: "var(--muted)" }}>
        {AGENTS.map((a) => (
          <li key={a.id} style={{ marginBottom: "0.15rem" }}>
            <strong>{a.name}</strong> ({a.role}): {a.desc}
          </li>
        ))}
      </ul>
      <button
        className="ghost"
        onClick={() => { try { localStorage.setItem("nexus_welcome_dismissed", "1"); } catch { } setDismissed(true); }}
        aria-label="Dismiss welcome"
        style={{ padding: "0.15rem 0.4rem", fontSize: "0.7rem", float: "right", marginTop: "-0.25rem" }}
      >
        ×
      </button>
    </section>
  );
}

export default function RootLayout({ children }) {
  const pathname = usePathname() || "/";

  // Visibility pause/resume is owned by motion.js (single listener there).

  return (
    <html lang="en">
      <head>
        <title>NEXUS — Multi-Agent Negotiation & Explainability</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <header className="topbar">
          <Link href="/" className="brand">NEXUS</Link>
          <nav>
            {NAV.map(([href, label]) => (
              <Link
                key={href}
                href={href}
                className={pathname === href ? "active" : undefined}
                aria-current={pathname === href ? "page" : undefined}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="topbar-right">
            <ThemeToggle />
          </div>
        </header>
        <div className="shell">
          <aside className="left-rail">
            <GuideChecklist />
            <WelcomePanel />
          </aside>
          <main>{children}</main>
        </div>
        <nav className="bottom-nav" aria-label="Bottom navigation">
          {NAV.map(([href, label]) => (
            <Link
              key={href}
              href={href}
              className={pathname === href ? "active" : undefined}
              aria-current={pathname === href ? "page" : undefined}
            >
              <span>{label.charAt(0)}</span>
              <span>{label}</span>
            </Link>
          ))}
        </nav>
      </body>
    </html>
  );
}
