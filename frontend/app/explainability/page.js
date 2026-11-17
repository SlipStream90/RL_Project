"use client";
import { useState } from "react";
import { api, API } from "../../lib/api";

const EXAMPLE_QUESTIONS = [
  { q: "Why did Buyer A reject Seller A's first offer?", hint: "First recorded REJECT by Buyer A, with reward components" },
  { q: "Why did Seller B counteroffer?", hint: "First COUNTEROFFER by Seller B and what it earned" },
  { q: "Why did Buyer B accept?", hint: "The ACCEPT that closed a deal, if one is recorded" },
  { q: "Who got the best deal?", hint: "Deal ranked by buyer saving vs reservation price" },
  { q: "How many deals were closed?", hint: "Every closed deal with buyer, seller, price, quantity" },
  { q: "What did the market manager do?", hint: "All manager actions and cumulative reward" },
  { q: "What is the reservation price for Seller A?", hint: "Private config for Seller A (₹90 by default)" },
  { q: "What is the reservation price for Buyer B?", hint: "Private config for Buyer B (₹110 by default)" },
  { q: "How was Buyer A rewarded?", hint: "Latest reward and its recorded components" },
  { q: "What happened at step 2?", hint: "The recorded action and reward at timestep 2" },
  { q: "Why did Seller A wait?", hint: "First WAIT by Seller A, or a plain 'not recorded' answer" },
  { q: "Why did Buyer A withdraw?", hint: "First WITHDRAW by Buyer A, or a plain 'not recorded' answer" },
];

const AGENT_OPTIONS = ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"];
const QUESTION_TYPES = [
  { id: "why", label: "Why this action?" },
  { id: "reservation", label: "What was the reservation price?" },
  { id: "reward", label: "How was it rewarded?" },
];

export default function Explain() {
  const [q, setQ] = useState(EXAMPLE_QUESTIONS[0].q);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [picker, setPicker] = useState({ agent: "", step: "", type: "why" });

  const [ep, setEp] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("nexus_episode") || undefined;
    return undefined;
  });

  const useExample = (question) => {
    setQ(question);
    ask(question, undefined);
  };

  const buildQuestion = () => {
    const { agent, step, type } = picker;
    if (type === "reservation") return `What is the reservation price for ${agent}?`;
    if (type === "reward") return `How was the agent ${agent} rewarded at step ${step || "the last step"}?`;
    if (type === "why") return `Why did ${agent} take that action at step ${step || "the last step"}?`;
    return q;
  };

  const ask = async (questionOverride, agentOverride) => {
    if (!ep) { setErr("No episode in storage. Start one in Marketplace first."); return; }
    setLoading(true);
    setErr("");
    const question = questionOverride ?? q;
    const agentId = agentOverride !== undefined ? agentOverride : (picker.agent || undefined);
    try {
      const r = await fetch(`${API}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, episode_id: ep, agent_id: agentId }),
      });
      if (!r.ok) throw new Error(`${r.status} query`);
      const data = await r.json();
      setRes(data);
      setQ(question);
      try { localStorage.setItem("nexus_queried", "1"); } catch { }
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  };

  const decodeActionEvidence = (e) => {
    const val = e.value;
    if (!val || !Array.isArray(val)) return String(val);
    const [atype, band, qty] = val;
    const typeName = ["OFFER", "COUNTEROFFER", "ACCEPT", "REJECT", "WAIT", "WITHDRAW"][atype] ?? String(atype);
    // Per quirk #1: price/qty come from active_offer/deals, not decoded from action array
    return `Chose ${typeName} (raw indices: type=${atype}, band=${band}, qty=${qty}). Price/qty come from state.active_offer, not decoded here.`;
  };

  const renderEvidence = (e) => {
    switch (e.type) {
      case "deal":
        return (
          <div className="evidence-item">
            <div className="meta">
              <span style={{ fontWeight: 600, color: "var(--ink)" }}>deal</span>
              <span style={{ color: "var(--muted)" }}>{e.field}</span>
              {e.timestep != null && <span className="mono">step {e.timestep}</span>}
            </div>
            <div className="mono">{typeof e.value === "object" && e.value ? `${e.value.buyer} bought from ${e.value.seller} at ₹${e.value.price} × ${e.value.quantity}` : String(e.value)}</div>
            <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "var(--muted)" }}>
              Source: {e.source}{e.episode_id ? ` · episode ${e.episode_id}` : ""}
            </div>
          </div>
        );
      case "action":
        return (
          <div className="evidence-item">
            <div className="meta">
              <span style={{ fontWeight: 600, color: "var(--ink)" }}>action</span>
              <span style={{ color: "var(--muted)" }}>{e.field}</span>
              {e.timestep != null && <span className="mono">step {e.timestep}</span>}
              {e.agent_id && <span className="mono">{e.agent_id}</span>}
            </div>
            <div className="mono">{decodeActionEvidence(e)}</div>
            <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "var(--muted)" }}>
              Source: {e.source}{e.episode_id ? ` · episode ${e.episode_id}` : ""}
            </div>
          </div>
        );
      case "reward_component":
        return (
          <div className="evidence-item">
            <div className="meta">
              <span style={{ fontWeight: 600, color: "var(--ink)" }}>reward_component</span>
              <span style={{ color: "var(--muted)" }}>{e.field}</span>
              {e.timestep != null && <span className="mono">step {e.timestep}</span>}
              {e.agent_id && <span className="mono">{e.agent_id}</span>}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
              <div style={{ width: "120px", height: 8, background: "var(--line)", borderRadius: "4px", overflow: "hidden" }}>
                <div style={{ height: "100%", width: Math.min(Math.abs(Number(e.value) || 0) * 100, 100), background: Number(e.value) >= 0 ? "var(--role-buyer-border)" : "var(--role-seller-border)" }} />
              </div>
              <span className="mono" style={{ minWidth: "60px" }}>{typeof e.value === "number" ? (e.value).toFixed(3) : String(e.value)}</span>
            </div>
            <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "var(--muted)" }}>
              Source: {e.source}{e.episode_id ? ` · episode ${e.episode_id}` : ""}
            </div>
          </div>
        );
      case "reservation_price":
        return (
          <div className="evidence-item">
            <div className="meta">
              <span style={{ fontWeight: 600, color: "var(--ink)" }}>reservation_price</span>
              <span style={{ color: "var(--muted)" }}>{e.field}</span>
              {e.agent_id && <span className="mono">{e.agent_id}</span>}
            </div>
            <div className="mono" style={{ marginTop: "0.25rem" }}>
              {typeof e.value === "number" ? `₹${e.value.toFixed(2)}` : String(e.value)}
            </div>
            <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "var(--muted)" }}>
              Source: {e.source}{e.episode_id ? ` · episode ${e.episode_id}` : ""}
            </div>
          </div>
        );
      default:
        return (
          <div className="evidence-item">
            <div className="meta">
              <span style={{ fontWeight: 600, color: "var(--ink)" }}>{e.type}</span>
              <span style={{ color: "var(--muted)" }}>{e.field}</span>
              {e.timestep != null && <span className="mono">step {e.timestep}</span>}
              {e.agent_id && <span className="mono">{e.agent_id}</span>}
            </div>
            <div className="mono" style={{ marginTop: "0.25rem" }}>{String(e.value)}</div>
            <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "var(--muted)" }}>
              Source: {e.source}{e.episode_id ? ` · episode ${e.episode_id}` : ""}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="dashboard-page" data-anim="page-transition">
      <h1>Explainability</h1>
      <p className="lede">Ask about a recorded decision. The answer formats trajectory evidence; it does not invent prices, probabilities, or rewards. Every number cited comes from the recorded trace.</p>

      {/* Structured picker */}
      <div className="panel">
        <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Structured question builder</h3>
        <p style={{ fontSize: "0.8rem", color: "var(--muted)", margin: "0 0 0.75rem 0" }}>
          Pick an agent, step, and question type. The question text is built automatically and sent with episode_id and agent_id.
        </p>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
          <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ink)" }}>Agent</label>
          <select value={picker.agent} onChange={(e) => setPicker({ ...picker, agent: e.target.value })} style={{ fontSize: "0.82rem", padding: "0.45rem 0.6rem" }}>
            <option value="">(any)</option>
            {AGENT_OPTIONS.map((a) => (
              <option key={a} value={a}>{a.replace("_", " ").toUpperCase()}</option>
            ))}
          </select>
          <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ink)" }}>Step</label>
          <input type="number" value={picker.step} onChange={(e) => setPicker({ ...picker, step: e.target.value })} style={{ fontSize: "0.82rem", padding: "0.45rem 0.6rem", width: "90px" }} />
          <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ink)" }}>Type</label>
          <select value={picker.type} onChange={(e) => setPicker({ ...picker, type: e.target.value })} style={{ fontSize: "0.82rem", padding: "0.45rem 0.6rem" }}>
            {QUESTION_TYPES.map((qt) => (
              <option key={qt.id} value={qt.id}>{qt.label}</option>
            ))}
          </select>
          <button onClick={() => ask(buildQuestion(), picker.agent || undefined)} style={{ marginLeft: "auto" }}>
            {loading ? "Asking…" : "Ask"}
          </button>
        </div>
        <p style={{ fontSize: "0.8rem", color: "var(--muted)", margin: 0 }}>
          Built question: <span className="mono">{buildQuestion()}</span>{picker.agent ? ` · agent: ${picker.agent}` : ""}
        </p>
      </div>

      {/* Question palette */}
      <div className="panel">
        <h3 style={{ marginTop: 0, fontSize: "1.05rem" }}>Question palette — tap one to ask it now</h3>
        <p style={{ fontSize: "0.82rem", color: "var(--muted)", margin: "0 0 0.85rem 0" }}>
          Each card asks the live backend about the recorded episode. Answers cite trajectory, deal, or config evidence only; if the event has not happened yet, the answer says so plainly.
        </p>
        <div className="palette-grid" data-testid="question-palette">
          {EXAMPLE_QUESTIONS.map(({ q: qq, hint }) => (
            <button key={qq} className="chip-btn" onClick={() => useExample(qq)} title={qq}>
              {qq}
              <small>{hint}</small>
            </button>
          ))}
        </div>
      </div>

      {/* Free text + submit */}
      <div className="panel">
        <div className="ask">
          <input aria-label="Question" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()} />
          <button onClick={() => ask()} disabled={loading}>{loading ? "Asking…" : "Ask"}</button>
        </div>
        {err && <div className="alert" role="alert">{err}. Start an episode and step it in Marketplace first.</div>}
        {loading && <div className="loading">Generating explanation from trajectory evidence…</div>}
        {res && (
          <div className="panel">
            <p style={{ fontSize: "0.95rem", lineHeight: 1.6 }}>{res.answer}</p>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
              <div style={{ width: "60px", height: "60px", borderRadius: "50%", background: "var(--line)", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                <svg width="52" height="52">
                  <circle cx="26" cy="26" r="22" fill="none" stroke="var(--line)" strokeWidth="3" />
                  <circle cx="26" cy="26" r="22" fill="none" stroke="var(--role-buyer-border)" strokeWidth="3"
                    style={{ strokeDasharray: "138", strokeDashoffset: 138 - (138 * (res.confidence || 0)), transform: "rotate(-90 26 26)", transition: "stroke-dashoffset 0.6s ease" }} />
                </svg>
                <span style={{ fontSize: "0.75rem", fontWeight: 600, fontFamily: "var(--font-mono)" }}>{((res.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
              <div>
                <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Confidence</div>
                <div style={{ fontSize: "0.85rem", fontWeight: 600, fontFamily: "var(--font-mono)" }}>{((res.confidence || 0) * 100).toFixed(1)}% <span style={{ fontSize: "0.7rem", color: "var(--muted)", fontWeight: 400 }}>(backend-reported)</span></div>
                <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Sources: {res.sources?.join(", ") || "—"}</div>
              </div>
            </div>
            <h3 style={{ fontSize: "1rem", marginTop: "1rem" }}>Evidence panel</h3>
            {res.evidence && res.evidence.length > 0 ? (
              <div>
                {res.evidence.map((e, i) => (
                  <div key={i}>{renderEvidence(e)}</div>
                ))}
              </div>
            ) : (
              <p className="empty">No evidence returned by the backend for this question.</p>
            )}
            {res.evidence && res.evidence.length <= 1 && (
              <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.5rem", fontStyle: "italic" }}>
                Thin evidence — the backend only returned a single evidence item for this question.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
