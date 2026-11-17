"use client";
import { useState, useEffect, useRef } from "react";
import { api, ACTION_NAMES, bandToPrice, agentRole, API } from "../../lib/api";

const AGENT_ORDER = ["buyer_a", "buyer_b", "market_manager", "seller_a", "seller_b"];

const NODE_POS = {
  buyer_a: { cx: 110, cy: 110 },
  buyer_b: { cx: 110, cy: 190 },
  market_manager: { cx: 220, cy: 50 },
  seller_a: { cx: 330, cy: 110 },
  seller_b: { cx: 330, cy: 190 },
};

function roleColor(role) {
  switch (role) {
    case "buyer": return "var(--role-buyer)";
    case "seller": return "var(--role-seller)";
    case "manager": return "var(--role-manager)";
    default: return "var(--ink)";
  }
}
function roleFill(role) {
  switch (role) {
    case "buyer": return "var(--role-buyer-fill)";
    case "seller": return "var(--role-seller-fill)";
    case "manager": return "var(--role-manager-fill)";
    default: return "var(--paper)";
  }
}
function roleBorder(role) {
  switch (role) {
    case "buyer": return "var(--role-buyer-border)";
    case "seller": return "var(--role-seller-border)";
    case "manager": return "var(--role-manager-border)";
    default: return "var(--ink)";
  }
}

function posFor(agentId) {
  if (!agentId || !NODE_POS[agentId]) return null;
  return NODE_POS[agentId];
}

export default function Replay() {
  const [rows, setRows] = useState([]);
  const [scrub, setScrub] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const [ep, setEp] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("nexus_episode") || "";
    return "";
  });

  useEffect(() => {
    if (!ep) return;
    setLoading(true);
    fetch(`${API}/api/v1/episodes/${ep}/trajectory`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} trajectory`);
        return r.json();
      })
      .then((data) => {
        setRows(data);
        setScrub(null);
        setErr("");
        try { localStorage.setItem("nexus_replayed", "1"); } catch { }
        setLoading(false);
      })
      .catch((e) => {
        setErr(`${e.message} — start an episode in Marketplace first.`);
        setLoading(false);
      });
  }, [ep]);

  const load = () => {
    if (!ep) { setErr("No episode in storage. Start one in Marketplace first."); return; }
    setLoading(true);
    fetch(`${API}/api/v1/episodes/${ep}/trajectory`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} trajectory`);
        return r.json();
      })
      .then((data) => {
        setRows(data);
        setScrub(null);
        setErr("");
        setLoading(false);
      })
      .catch((e) => {
        setErr(`${e.message} — start an episode in Marketplace first.`);
        setLoading(false);
      });
  };

  const selected = scrub != null ? rows.find((t) => t.timestep === scrub) : null;
  const scrubParticleRef = useRef(null);

  // Scrub-highlight: replay the message particle from the scrubbed actor toward the hub.
  useEffect(() => {
    const el = scrubParticleRef.current;
    if (!el || !selected) return;
    const from = posFor(selected.agent_id);
    if (!from) return;
    const hub = NODE_POS.market_manager;
    const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const a = el.animate(
      reduced ? [{ opacity: 0 }, { opacity: 1 }, { opacity: 0 }] : [
        { transform: "translate(0px, 0px)", opacity: 1 },
        { transform: `translate(${hub.cx - from.cx}px, ${hub.cy - from.cy}px)`, opacity: 0.85 },
      ],
      { duration: reduced ? 240 : 700, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" }
    );
    return () => { try { a.cancel(); } catch { /* noop */ } };
  }, [selected]);

  const roleOf = (aid) => {
    if (aid.startsWith("buyer")) return "buyer";
    if (aid.startsWith("seller")) return "seller";
    return "manager";
  };

  return (
    <div className="dashboard-page" data-anim="page-transition">
      <h1>Negotiation Replay</h1>
      <p className="lede">Turn-by-turn record for the current episode. The current episode loads automatically; use the scrubber to jump to any turn and see it highlighted on the network.</p>

      <div className="panel">
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center", marginBottom: "0.75rem" }}>
          <button onClick={load}>Refresh replay</button>
          <span className="mono">Episode: {ep || "—"}</span>
          <span className="mono">Turns: {rows.length}</span>
          {scrub != null && <span className="mono">Scrubbed to step {scrub}</span>}
        </div>
        {err && <div className="alert" role="alert">{err}</div>}
        {loading && <div className="loading">Loading trajectory…</div>}
      </div>

      <div className="panel">
        <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Network — scrubbed turn</h3>
        {rows.length === 0 ? (
          <p className="empty">No turns loaded yet. Start and step an episode in Marketplace.</p>
        ) : (
          <div style={{ position: "relative", width: "100%", height: 150, border: "1px solid var(--line)", borderRadius: "var(--radius-sm)", background: "var(--paper)" }}>
            <svg width="100%" height="100%" role="img" aria-label="Replay network highlight">
              <line x1="110" y1="110" x2="220" y2="50" stroke="var(--line)" strokeWidth="1" />
              <line x1="110" y1="190" x2="220" y2="50" stroke="var(--line)" strokeWidth="1" />
              <line x1="330" y1="110" x2="220" y2="50" stroke="var(--line)" strokeWidth="1" />
              <line x1="330" y1="190" x2="220" y2="50" stroke="var(--line)" strokeWidth="1" />
              <line x1="110" y1="110" x2="330" y2="110" stroke="var(--line)" strokeWidth="1" />
              <line x1="110" y1="190" x2="330" y2="190" stroke="var(--line)" strokeWidth="1" />

              {selected && selected.agent_id && posFor(selected.agent_id) && (
                <>
                  <circle ref={scrubParticleRef} data-anim="scrub-highlight" cx={NODE_POS[selected.agent_id].cx} cy={NODE_POS[selected.agent_id].cy} r="6" fill={roleColor(roleOf(selected.agent_id))} style={{ transformBox: "fill-box", transformOrigin: "center" }} />
                  <line
                    x1={NODE_POS[selected.agent_id].cx}
                    y1={NODE_POS[selected.agent_id].cy}
                    x2={selected.action?.[0] === 2 || selected.action?.[0] === 3 ? 220 : NODE_POS[selected.agent_id].cx}
                    y2={selected.action?.[0] === 2 || selected.action?.[0] === 3 ? 50 : NODE_POS[selected.agent_id].cy - 30}
                    stroke={roleColor(roleOf(selected.agent_id))}
                    strokeWidth="3"
                    strokeDasharray="6,4"
                    style={{ animation: "draw 0.6s ease forwards" }}
                  />
                </>
              )}

              {AGENT_ORDER.map((aid) => {
                const isHighlight = selected && selected.agent_id === aid;
                const pos = NODE_POS[aid];
                return (
                  <g key={aid}>
                    <circle cx={pos.cx} cy={pos.cy} r={isHighlight ? 10 : 7}
                      fill={roleFill(roleOf(aid))} stroke={roleBorder(roleOf(aid))}
                      strokeWidth={isHighlight ? 3 : 2} />
                    <text x={pos.cx} y={pos.cy + 3} textAnchor="middle"
                      fontSize="9" fontWeight="600" fill="var(--ink)" fontFamily="var(--font-sans)" pointerEvents="none">
                      {aid.replace("_", " ").toUpperCase()}
                    </text>
                  </g>
                );
              })}
              {selected && (
                <text x="160" y="140" textAnchor="middle" fontSize="10" fill="var(--muted)" fontFamily="var(--font-sans)">
                  {selected.agent_id} · {ACTION_NAMES[selected.action?.[0]] ?? selected.action?.[0]}
                </text>
              )}
            </svg>
            <p style={{ margin: "0.4rem 0 0", fontSize: "0.8rem", color: "var(--muted)", textAlign: "center" }}>
              {selected ? `Step {selected.timestep}: {selected.agent_id} {ACTION_NAMES[selected.action?.[0]] ?? selected.action?.[0]}` : "Scrub to select a turn"}
            </p>
          </div>
        )}
      </div>

      <div className="panel">
        <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Turn scrubber</h3>
        {rows.length === 0 ? (
          <p className="empty">No turns loaded yet. Start and step an episode in Marketplace.</p>
        ) : (
          <>
            <input type="range" min={0} max={Math.max(rows.length - 1, 0)} value={scrub ?? 0} step={1} onChange={(e) => setScrub(rows[Number(e.target.value)]?.timestep ?? Number(e.target.value))} aria-label="Select replay turn" />
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-sans)", fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.25rem" }}>
              <span>Step {rows[0]?.timestep ?? 0}</span>
              <span>Step {rows[rows.length - 1]?.timestep ?? 0}</span>
            </div>
            {selected && (
              <div style={{ marginTop: "0.75rem", padding: "0.75rem", background: "var(--paper)", border: "1px solid var(--line)", borderRadius: "var(--radius-sm)" }}>
                <p style={{ margin: 0, fontFamily: "var(--font-sans)", fontSize: "0.82rem" }}>
                  <span className={`chip ${roleOf(selected.agent_id)}`}>{selected.agent_id}</span>{" "}
                  <span className={`action-badge ${(ACTION_NAMES[selected.action[0]] ?? "").toLowerCase()}`}>{ACTION_NAMES[selected.action[0]] ?? selected.action[0]}</span>
                  {selected.action[1] != null && <span className="mono">band idx {selected.action[1]}</span>}
                  {selected.action[2] != null && <span className="mono">qty idx {selected.action[2]}</span>}
                  <span className={`reward-badge ${selected.reward > 0 ? "pos" : selected.reward < 0 ? "neg" : "zero"}`}>{selected.reward.toFixed(3)}</span>
                </p>
                {selected.utterance && <p style={{ margin: "0.35rem 0 0", fontSize: "0.82rem", color: "var(--muted)" }}>{selected.utterance}</p>}
              </div>
            )}
          </>
        )}
      </div>

      <div className="panel">
        <table aria-label="Replay">
          <thead>
            <tr><th>Step</th><th>Agent</th><th>Action</th><th>Band / Qty</th><th>Reward</th></tr>
          </thead>
          <tbody>
            {rows.map((t) => (
              <tr key={t.timestep} style={{ background: scrub === t.timestep ? "rgba(15, 107, 79, 0.15)" : "inherit" }}>
                <td>{t.timestep}</td>
                <td><span className={`chip ${roleOf(t.agent_id)}`}>{t.agent_id}</span></td>
                <td>{ACTION_NAMES[t.action?.[0]] ?? t.action?.[0]}</td>
                <td className="mono">{t.action?.[1] ?? "—"} / {t.action?.[2] ?? "—"}</td>
                <td className={`reward-badge ${t.reward > 0 ? "pos" : t.reward < 0 ? "neg" : "zero"}`}>{Number(t.reward).toFixed(3)}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={5} className="empty">No turns loaded yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
