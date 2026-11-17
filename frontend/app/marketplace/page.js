"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { api, ACTION_NAMES, agentRole, API } from "../../lib/api";
import {
  particleManager,
  stepRequester,
  pauseAnimations,
  resumeAnimations,
  onResume,
  scaledDuration,
} from "../../app/motion";

const AGENT_ORDER = ["buyer_a", "buyer_b", "market_manager", "seller_a", "seller_b"];

const NODE_POS = {
  buyer_a: { cx: 120, cy: 220 },
  buyer_b: { cx: 120, cy: 380 },
  market_manager: { cx: 500, cy: 120 },
  seller_a: { cx: 880, cy: 220 },
  seller_b: { cx: 880, cy: 380 },
};

const EDGES = [
  { from: "buyer_a", to: "market_manager" },
  { from: "buyer_b", to: "market_manager" },
  { from: "seller_a", to: "market_manager" },
  { from: "seller_b", to: "market_manager" },
  { from: "buyer_a", to: "seller_a" },
  { from: "buyer_b", to: "seller_b" },
];

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

export default function Marketplace() {
  const [ep, setEp] = useState("");
  const [state, setState] = useState(null);
  const [prevState, setPrevState] = useState(null);
  const [rewards, setRewards] = useState({});
  const [turns, setTurns] = useState([]);
  const [err, setErr] = useState("");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [anim, setAnim] = useState(null);
  const [offerPrices, setOfferPrices] = useState([]);
  const [researchView, setResearchView] = useState(false);
  const [seed, setSeed] = useState(42);
  const [maxSteps, setMaxSteps] = useState(100);
  const intervalRef = useRef(null);
  const prevStateRef = useRef(null);
  const steppingRef = useRef(false);
  const playingRef = useRef(false);
  const speedRef = useRef(1);
  const particleRef = useRef(null);
  const agentConfigsRef = useRef(null);
  const [agentConfigs, setAgentConfigs] = useState(null);

  // Persist episode id for other pages
  useEffect(() => {
    if (ep) {
      try { localStorage.setItem("nexus_episode", ep); } catch { }
      window.dispatchEvent(new CustomEvent("nexus-episode-changed"));
    }
  }, [ep]);

  useEffect(() => {
    try { localStorage.setItem("nexus_turns", String(turns.length)); } catch { }
  }, [turns.length]);

  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { playingRef.current = playing; }, [playing]);

  // Fetch agent configs (reservation prices) for research view
  useEffect(() => {
    api("/api/v1/agents")
      .then((data) => { agentConfigsRef.current = data; setAgentConfigs(data); })
      .catch(() => { /* configs optional */ });
  }, []);

  // Animate the declarative msg-particle (WAAPI, transform only) whenever anim changes.
  useEffect(() => {
    if (!anim || !particleRef.current) return;
    const el = particleRef.current;
    if (anim.kind === "wait" || anim.kind === "withdraw") return;
    const fromPos = posFor(anim.from);
    const toPos = anim.to ? posFor(anim.to) : null;
    if (!fromPos || !toPos) return;
    const dx = toPos.cx - fromPos.cx;
    const dy = toPos.cy - fromPos.cy;
    const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const a = el.animate(
      reduced
        ? [{ opacity: 0 }, { opacity: 1 }, { opacity: 0 }]
        : [
            { transform: "translate(0px, 0px)", opacity: 1 },
            { transform: `translate(${dx}px, ${dy}px)`, opacity: 0.85 },
          ],
      { duration: reduced ? 240 : anim.travelMs, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" }
    );
    particleManager.register(`msg-${anim.from}-${anim.to}`, a);
    return () => { try { a.cancel(); } catch { /* noop */ } };
  }, [anim]);

  const clearAll = useCallback(() => {
    if (intervalRef.current) { clearTimeout(intervalRef.current); intervalRef.current = null; }
    particleManager.cancelAll();
    stepRequester.cancelAll();
  }, []);

  const start = async () => {
    clearAll();
    playingRef.current = false;
    setPlaying(false);
    try {
      const r = await api("/api/v1/simulation/start", {
        method: "POST",
        body: JSON.stringify({ seed: Number(seed) || 42, max_steps: Number(maxSteps) || 100 }),
      });
      prevStateRef.current = null;
      setEp(r.episode_id);
      setState(r.state);
      setPrevState(null);
      setRewards({});
      setTurns([]);
      setOfferPrices([]);
      setErr("");
      setAnim(null);
    } catch (e) { setErr(e.message); }
  };

  const handleStepResult = (r) => {
    const { agent, action, reward, state: st, done } = r;
    const prev = prevStateRef.current;
    if (!agent) {
      prevStateRef.current = st;
      setPrevState(prev);
      setState(st);
      if (done) { playingRef.current = false; setPlaying(false); }
      return;
    }
    const oldOffer = prev ? prev.active_offer : null;
    const newOffer = st.active_offer;
    prevStateRef.current = st;
    setPrevState(prev);
    setState(st);
    setRewards((p) => ({ ...p, [agent]: (p[agent] || 0) + (reward ?? 0) }));

    if (newOffer && (!oldOffer || oldOffer.price !== newOffer.price || oldOffer.quantity !== newOffer.quantity || oldOffer.offerer !== newOffer.offerer)) {
      setOfferPrices((p) => [...p, { step: st.step, price: newOffer.price, quantity: newOffer.quantity, offerer: newOffer.offerer }]);
    }
    const oldDeals = prev && Array.isArray(prev.deals) ? prev.deals : [];
    const newDeals = Array.isArray(st.deals) ? st.deals : [];
    const dealAdded = newDeals.length > oldDeals.length;
    const latestDeal = dealAdded ? newDeals[newDeals.length - 1] : null;

    const atype = action?.[0];
    const typeName = ACTION_NAMES[atype] ?? String(atype);
    let to = null;
    let kind = "offer";
    let price = null;
    let qty = null;
    if (atype === 4) { kind = "wait"; }
    else if (atype === 5) { kind = "withdraw"; price = oldOffer ? oldOffer.price : null; qty = oldOffer ? oldOffer.quantity : null; }
    else if (atype === 0 || atype === 1) { kind = "offer"; to = "market_manager"; price = newOffer ? newOffer.price : null; qty = newOffer ? newOffer.quantity : null; }
    else if (atype === 2 || atype === 3) { kind = atype === 2 ? "accept" : "reject"; to = oldOffer ? oldOffer.offerer : null; price = oldOffer ? oldOffer.price : null; qty = oldOffer ? oldOffer.quantity : null; }

    const travelMs = scaledDuration(700, speedRef.current);
    setAnim({ from: agent, to, kind, actionType: atype, label: typeName, price, qty, deal: latestDeal, travelMs });
    setTurns((p) => [{ agent, action, reward: reward ?? 0, kind, label: typeName, price, qty, ts: Date.now() }, ...p].slice(0, 12));
    if (done) { playingRef.current = false; setPlaying(false); }
  };

  const stepOnce = async () => {
    if (!ep || steppingRef.current) return false;
    steppingRef.current = true;
    try {
      const res = await fetch(`${API}/api/v1/simulation/${ep}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: null }),
      });
      if (!res.ok) throw new Error(`${res.status} /step`);
      const r = await res.json();
      handleStepResult(r);
      setErr("");
      return !r.done;
    } catch (e) {
      setErr(e.message);
      playingRef.current = false;
      setPlaying(false);
      return false;
    } finally {
      steppingRef.current = false;
    }
  };

  const reset = () => {
    clearAll();
    playingRef.current = false;
    setPlaying(false);
    prevStateRef.current = null;
    setEp("");
    setState(null);
    setPrevState(null);
    setRewards({});
    setTurns([]);
    setOfferPrices([]);
    setAnim(null);
    setErr("");
    try { localStorage.removeItem("nexus_episode"); localStorage.setItem("nexus_turns", "0"); } catch { /* noop */ }
  };

  const togglePlay = () => {
    if (!ep) return;
    if (playingRef.current) {
      playingRef.current = false;
      setPlaying(false);
      if (intervalRef.current) { clearTimeout(intervalRef.current); intervalRef.current = null; }
      return;
    }
    playingRef.current = true;
    setPlaying(true);
    const loop = async () => {
      if (!playingRef.current) return;
      const more = await stepOnce();
      if (!playingRef.current || !more) { playingRef.current = false; setPlaying(false); return; }
      intervalRef.current = setTimeout(loop, scaledDuration(500, speedRef.current));
    };
    loop();
  };

  useEffect(() => {
    return clearAll;
  }, [clearAll]);

  const navigateToProfile = (agentId) => {
    if (typeof window !== "undefined") {
      window.location.href = `/agent-profiles?agent=${encodeURIComponent(agentId)}`;
    }
  };

  const offerers = [...new Set(offerPrices.map((p) => p.offerer))];
  const maxStep = state ? state.step : 0;
  const maxPrice = offerPrices.length > 0 ? Math.max(...offerPrices.map((p) => p.price)) : 200;

  const resByAgent = {};
  if (agentConfigs) {
    agentConfigs.forEach((c) => { resByAgent[c.agent_id] = c.reservation_price; });
  }

  let dealZone = null;
  if (researchView && agentConfigs) {
    const buyers = agentConfigs.filter((c) => c.role === "buyer");
    const sellers = agentConfigs.filter((c) => c.role === "seller");
    if (buyers.length && sellers.length) {
      const minBuyerRes = Math.max(...buyers.map((b) => b.reservation_price));
      const maxSellerRes = Math.min(...sellers.map((s) => s.reservation_price));
      dealZone = { minBuyerRes, maxSellerRes, viable: minBuyerRes >= maxSellerRes };
    }
  }

  const fromPos = anim && posFor(anim.from);
  const toPos = anim && anim.to && posFor(anim.to);

  // KPI values
  const totalReward = Object.values(rewards).reduce((a, b) => a + b, 0);
  const lastOffer = offerPrices.length > 0 ? offerPrices[offerPrices.length - 1] : null;

  return (
    <div className="dashboard-page" data-anim="page-transition">
      <h1>Marketplace</h1>
      <p className="lede">Live five-agent marketplace. Rule-based baseline policies act on each turn (trained MAPPO/IPPO numbers live in Experiment Lab). Sellers concede toward their floor and buyers raise bids over time, so prices evolve instead of repeating. Watch actions travel across the network, then inspect any agent’s private profile.</p>

      {err && <div className="alert" role="alert">{err}. Is the backend running on port 8000?</div>}

      {!ep ? (
        <div className="panel">
          <div className="controls">
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.8rem", color: "var(--muted)", fontWeight: 600 }}>
              Seed
              <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.8rem", color: "var(--muted)", fontWeight: 600 }}>
              Max steps
              <input type="number" value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} />
            </label>
            <button onClick={start}>Start episode</button>
          </div>
          <p className="empty">No episode running. Set seed/max steps and click “Start episode” to begin.</p>
        </div>
      ) : (
        <>
          <div className="panel dashboard-grid">
            <div className="dashboard-main">
              {/* KPI Tile Row */}
              <div className="kpi-row">
                <div className="kpi-tile">
                  <div className="kpi-label">Step</div>
                  <div className="kpi-value tabular-nums">{state.step}</div>
                  <div className="kpi-sub">of {maxSteps}</div>
                </div>
                <div className="kpi-tile">
                  <div className="kpi-label">Trades closed</div>
                  <div className="kpi-value tabular-nums">{state.deals.length}</div>
                  <div className="kpi-sub">completed deals</div>
                </div>
                <div className="kpi-tile">
                  <div className="kpi-label">Active offer</div>
                  <div className="kpi-value tabular-nums">
                    {state.active_offer ? `₹${state.active_offer.price} × ${state.active_offer.quantity}` : "—"}
                  </div>
                  <div className="kpi-sub">{state.active_offer ? state.active_offer.offerer : "none"}</div>
                </div>
                <div className="kpi-tile">
                  <div className="kpi-label">Last price</div>
                  <div className="kpi-value tabular-nums">
                    {lastOffer ? `₹${lastOffer.price}` : "—"}
                  </div>
                  <div className="kpi-sub">per unit</div>
                  {offerPrices.length > 1 && (() => {
                    const prices = offerPrices.map((p) => p.price);
                    const mn = Math.min(...prices); const mx = Math.max(...prices); const rg = Math.max(mx - mn, 1);
                    const pts = prices.map((v, i) => `${(i / (prices.length - 1)) * 100},${26 - ((v - mn) / rg) * 22}`).join(" ");
                    return (
                      <svg className="kpi-sparkline" viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden="true">
                        <polyline data-anim="kpi-sparkline" pathLength={1} points={pts} fill="none" stroke="var(--role-buyer)" strokeWidth="2" />
                      </svg>
                    );
                  })()}
                </div>
                <div className="kpi-tile">
                  <div className="kpi-label">Total reward</div>
                  <div className="kpi-value tabular-nums">{totalReward.toFixed(3)}</div>
                  <div className="kpi-sub">per agent, summed</div>
                </div>
              </div>

              <div className="controls header-controls" style={{ marginBottom: "0.75rem" }}>
                <span className="mono" style={{ fontWeight: 600 }}>Episode: {ep}</span>
                <span className="mono" style={{ marginLeft: "0.75rem" }}>Step {state.step} of {maxSteps}</span>
                <span className="mono" style={{ marginLeft: "0.75rem" }}>Up next: {state.agent_selection ? state.agent_selection : "done"}</span>
                <label className="research-toggle">
                  <input type="checkbox" checked={researchView} onChange={(e) => setResearchView(e.target.checked)} />
                  Show reservation prices
                </label>
              </div>

              {/* Network graph */}
              <div className="net-wrap">
                <svg className="net-svg" viewBox="0 0 1000 440" aria-label="Agent negotiation network" role="group">
                  <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L9,3 z" fill="var(--muted)" />
                    </marker>
                    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                  </defs>

                  {researchView && dealZone && dealZone.viable && (
                    <rect x="0" y="0" width="1000" height="440" fill="var(--role-seller)" opacity="0.05" />
                  )}

                  {EDGES.map((e, i) => {
                    const a = NODE_POS[e.from]; const b = NODE_POS[e.to];
                    const mx = (a.cx + b.cx) / 2; const my = (a.cy + b.cy) / 2 - 30;
                    const used = anim && ((anim.from === e.from && anim.to === e.to) || (anim.from === e.to && anim.to === e.from));
                    return <path key={i} data-anim={used ? "edge-flow" : undefined} className={used ? "edge-flow-active" : "edge-static"} d={`M${a.cx},${a.cy} Q${mx},${my} ${b.cx},${b.cy}`} fill="none" />;
                  })}

                  {anim && fromPos && (
                    <g data-testid="particle-layer">
                      {(anim.kind === "offer" || anim.kind === "accept" || anim.kind === "reject") && toPos && (
                        <>
                          <path d={`M${fromPos.cx},${fromPos.cy} Q${(fromPos.cx + toPos.cx) / 2},${(fromPos.cy + toPos.cy) / 2 - 30} ${toPos.cx},${toPos.cy}`} fill="none" stroke={anim.kind === "accept" ? "var(--status-accept)" : anim.kind === "reject" ? "var(--status-reject)" : roleColor(agentRole(anim.from))} strokeWidth="2" opacity="0.35" />
                          <circle ref={particleRef} data-anim="msg-particle" data-from={anim.from} data-to={anim.to} data-kind={anim.kind} cx={fromPos.cx} cy={fromPos.cy} r="7" fill={anim.kind === "accept" ? "var(--status-accept)" : anim.kind === "reject" ? "var(--status-reject)" : roleColor(agentRole(anim.from))} style={{ transformBox: "fill-box", transformOrigin: "center" }} />
                          <circle data-anim="offer-ripple" cx={fromPos.cx} cy={fromPos.cy} r="14" fill="none" stroke={roleColor(agentRole(anim.from))} strokeWidth="2" opacity="0.5" />
                          {(agentRole(anim.from) === "buyer" ? ["seller_a", "seller_b"] : agentRole(anim.from) === "seller" ? ["buyer_a", "buyer_b"] : []).map((s) => (
                            <circle key={s} data-anim="offer-ripple" cx={NODE_POS[s].cx} cy={NODE_POS[s].cy} r="12" fill="none" stroke={roleColor(agentRole(anim.from))} strokeWidth="1.5" opacity="0.35" />
                          ))}
                        </>
                      )}
                      {anim.kind === "wait" && (
                        <text data-anim="wait-marker" x={fromPos.cx} y={fromPos.cy - 26} textAnchor="middle" fontSize="11" fill="var(--muted)">waiting…</text>
                      )}
                      {anim.kind === "withdraw" && (
                        <text data-anim="withdraw-marker" x={fromPos.cx} y={fromPos.cy - 26} textAnchor="middle" fontSize="11" fill="var(--status-withdraw)">offer withdrawn</text>
                      )}
                      {anim.deal && (
                        <g data-anim="deal-burst"><circle cx={500} cy={120} r="22" fill="none" stroke="var(--status-accept)" strokeWidth="3" /><text x={500} y={124} textAnchor="middle" fontSize="12" fill="var(--status-accept)">deal</text></g>
                      )}
                    </g>
                  )}

                  {AGENT_ORDER.map((aid) => {
                    const role = agentRole(aid);
                    const isActive = state.agent_selection === aid;
                    const pos = NODE_POS[aid];
                    const rewardsVal = rewards[aid] ?? 0;
                    const isAnimSource = anim?.from === aid;
                    return (
                      <g key={aid} style={{ cursor: "pointer" }} role="button" tabIndex={0}
                        onKeyDown={(e) => e.key === "Enter" && navigateToProfile(aid)}
                        onClick={() => navigateToProfile(aid)}
                        aria-label={`${aid}, ${role}, reward ${rewardsVal.toFixed(2)}, ${isActive ? "up next" : "idle"}`}>
                        <circle cx={pos.cx} cy={pos.cy} r={isAnimSource ? 18 : 16} fill={roleFill(role)} stroke={roleBorder(role)}
                          strokeWidth={isActive ? 4 : 3}
                          style={{ transition: `r ${scaledDuration(200, speed)}ms, filter ${scaledDuration(200, speed)}ms`, filter: isAnimSource ? "url(#glow)" : "none" }} />
                        <text className="node-label" x={pos.cx} y={pos.cy + 5}>{aid.replace("_", " ").toUpperCase()}</text>
                        <text className="node-reward" data-anim="reward-tick" x={pos.cx} y={pos.cy + 20}>reward {rewardsVal.toFixed(2)}</text>
                        {isActive && (
                          <circle data-anim="node-active" cx={pos.cx} cy={pos.cy - 22} r="5" fill="var(--role-buyer-border)" />
                        )}
                      </g>
                    );
                  })}
                </svg>

                {anim && fromPos && (
                  <div style={{ position: "absolute", top: "10px", left: "50%", transform: "translateX(-50%)", zIndex: 5, pointerEvents: "none" }}>
                    <span className={`action-badge ${anim.label.toLowerCase()}`} style={{ fontSize: "0.75rem" }}>
                      {anim.label}
                      {anim.price != null ? ` · ₹${anim.price}` : ""}
                      {anim.qty != null ? ` ×${anim.qty}` : ""}
                      {anim.deal && ` · deal: ${anim.deal.buyer}↔${anim.deal.seller} @₹${anim.deal.price}`}
                    </span>
                  </div>
                )}
              </div>

              {/* Price chart */}
              <div className="chart-wrap">
                <h3 style={{ margin: "0 0 0.6rem 0", fontSize: "0.95rem" }}>Offer prices over steps</h3>
                <svg className="chart-svg" viewBox="0 0 700 200" role="img" aria-label="Offer price chart">
                  <defs>
                    <clipPath id="chartClip"><rect width="640" height="140" /></clipPath>
                  </defs>
                  <g className="chart-axis">
                    <line x1="40" y1="180" x2="680" y2="180" stroke="var(--line)" strokeWidth="1" />
                    <line x1="40" y1="40" x2="40" y2="180" stroke="var(--line)" strokeWidth="1" />
                    {[50, 100, 150, 200].map((p) => {
                      const y = 180 - ((p - 50) / 150) * 140;
                      return (
                        <g key={p}>
                          <line x1="36" y1={y} x2="40" y2={y} stroke="var(--line)" strokeWidth="1" />
                          <text x="30" y={y + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="var(--font-sans)">{p}</text>
                        </g>
                      );
                    })}
                  </g>
                  <g clipPath="url(#chartClip)">
                    {offerers.map((offerer) => {
                      const pts = offerPrices.filter((p) => p.offerer === offerer);
                      if (pts.length < 2) return null;
                      const d = pts.map((p, i) => {
                        const x = 40 + (p.step / Math.max(maxStep, 1)) * 640;
                        const y = 180 - ((p.price - 50) / 150) * 140;
                        return i === 0 ? `M${x},${y}` : `L${x},${y}`;
                      }).join(" ");
                      return (
                        <path key={offerer} data-anim="chart-draw" pathLength={1} d={d} className="chart-line" stroke={roleColor(agentRole(offerer))} />
                      );
                    })}
                    {state?.deals?.map((d, i) => {
                      const x = 40 + (((d.timestep ?? 0) + 1) / Math.max(maxStep, 1)) * 640;
                      const y = 180 - (((d.price ?? 50) - 50) / 150) * 140;
                      return (
                        <circle key={i} className="chart-deal" cx={x} cy={y} />
                      );
                    })}
                    {researchView && agentConfigs && agentConfigs.filter((c) => c.reservation_price > 0).map((c) => {
                      const y = 180 - ((c.reservation_price - 50) / 150) * 140;
                      return <line key={c.agent_id} data-anim="chart-draw" x1="40" y1={y} x2="680" y2={y} stroke={roleColor(c.role === "buyer" ? "buyer" : c.role === "seller" ? "seller" : "manager")} strokeDasharray="5,4" strokeWidth="1.5" opacity="0.8" />;
                    })}
                    {researchView && dealZone && dealZone.viable && (() => {
                      const y1 = 180 - ((dealZone.minBuyerRes - 50) / 150) * 140;
                      const y2 = 180 - ((dealZone.maxSellerRes - 50) / 150) * 140;
                      return <rect data-testid="deal-zone" x="40" y={Math.min(y1, y2)} width="640" height={Math.abs(y2 - y1)} fill="var(--status-accept)" opacity="0.12" />;
                    })()}
                  </g>
                </svg>
                <div className="chart-legend">
                  {offerers.map((o) => (
                    <span key={o} className="item">
                      <span className="dot" style={{ background: roleColor(agentRole(o)) }} />
                      {o.replace("_", " ").toUpperCase()}
                    </span>
                  ))}
                </div>
                <p style={{ fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.4rem" }}>
                  Lines = offer prices per agent (from active_offer). Circles = completed deals (from state.deals).
                </p>
              </div>

              {/* Controls */}
              <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button className="secondary" onClick={stepOnce} disabled={!state || state.agent_selection === null}>Step once</button>
                <button onClick={togglePlay} className={playing ? "secondary pulse" : undefined}>{playing ? "Pause" : "Auto-play"}</button>
                <label className="speed-control">
                  Speed
                  <input type="range" min="0.5" max="4" step="0.5" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
                </label>
                <button onClick={reset}>Reset</button>
              </div>

              {state && !state.agent_selection && (
                <div className="episode-summary">
                  <h3>Episode complete</h3>
                  <p style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
                    Total deals: {state.deals.length} · Final step: {state.step}
                  </p>
                  <div className="deals-grid">
                    {state.deals.map((d, i) => (
                      <div key={i} className="deal-card">
                        <div className="roles">{d.buyer} ↔ {d.seller}</div>
                        <div style={{ fontWeight: 600 }}>{d.price} × {d.quantity}</div>
                        <div className="mono" style={{ fontSize: "0.75rem" }}>Step {d.timestep}</div>
                      </div>
                    ))}
                    {!state.deals.length && <p className="empty" style={{ padding: "0.5rem 0" }}>No deals completed.</p>}
                  </div>
                  <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem" }}>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Final budgets</div>
                      <ul style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", margin: "0.25rem 0 0 1rem", color: "var(--muted)" }}>
                        {Object.entries(state.budgets).map(([a, v]) => (
                          <li key={a}>{a}: {v}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Final inventories</div>
                      <ul style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", margin: "0.25rem 0 0 1rem", color: "var(--muted)" }}>
                        {Object.entries(state.inventories).map(([a, v]) => (
                          <li key={a}>{a}: {v}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Total reward per agent</div>
                      <ul style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", margin: "0.25rem 0 0 1rem", color: "var(--muted)" }}>
                        {AGENT_ORDER.map((a) => (
                          <li key={a}>{a}: {(rewards[a] ?? 0).toFixed(3)}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="dashboard-sidebar">
              <div className="panel">
                <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Live feed</h3>
                {turns.length === 0 ? (
                  <p className="empty">No turns yet. Step the simulation.</p>
                ) : (
                  <ul className="feed">
                    {turns.map((t, i) => (
                      <li key={i} style={{ animationDelay: `${i * 0.05}s` }}>
                        <span className={`chip ${agentRole(t.agent)}`}>{t.agent}</span>
                        <span className={`action-badge ${t.label.toLowerCase()}`}>{t.label}</span>
                        {t.price != null && <span className="mono">₹{t.price}</span>}
                        {t.qty != null && <span className="mono">×{t.qty}</span>}
                        <span className={`reward-badge ${t.reward > 0 ? "pos" : t.reward < 0 ? "neg" : "zero"}`}>{t.reward.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
