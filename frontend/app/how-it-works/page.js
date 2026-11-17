"use client";
import { useState, useEffect } from "react";
import Link from "next/link";

const AGENT_DESC = {
  buyer_a: { name: "Buyer A", role: "Purchaser", reservation: 85, desc: "Price-focused buyer — wants resources cheaply, stays below reservation price ₹85." },
  buyer_b: { name: "Buyer B", role: "Purchaser", reservation: 110, desc: "Speed-focused buyer — wants resources quickly, accepts faster to close deals." },
  seller_a: { name: "Seller A", role: "Supplier", reservation: 90, desc: "Margin-focused seller — maximizes profit per unit, won't sell below unit cost ₹60." },
  seller_b: { name: "Seller B", role: "Supplier", reservation: 80, desc: "Volume-focused seller — maximizes units sold, accepts lower prices to move inventory." },
  market_manager: { name: "Market Manager", role: "Controller", reservation: 0, desc: "Market stability — keeps price variance low and transaction rate high." },
};

export default function HowItWorks() {
  const [buyerRes, setBuyerRes] = useState(85);
  const [sellerRes, setSellerRes] = useState(90);

  const dealZone = buyerRes >= sellerRes;
  const zoneWidth = dealZone ? buyerRes - sellerRes : 0;

  // Scroll-reveal: IntersectionObserver, fires once per section.
  useEffect(() => {
    const els = document.querySelectorAll('[data-anim="scroll-reveal"]');
    if (!("IntersectionObserver" in window)) { els.forEach((el) => el.classList.add("revealed")); return; }
    els.forEach((el) => el.classList.add("reveal-pending"));
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { e.target.classList.add("revealed"); io.unobserve(e.target); }
      }
    }, { threshold: 0.12 });
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  return (
    <div className="dashboard-page" data-anim="page-transition">
      <h1>How NEXUS works</h1>
      <p className="lede">A multi-agent negotiation and explainability platform. Five RL agents bargain in a marketplace; every step is recorded so you can replay and question it. Content is drawn from <code>NEXUS_ENV_SPEC.md</code> and <code>CONTRACT.md</code>.</p>

      {/* Section 1: Marketplace and hidden information */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>1. The marketplace and hidden information</h2>
        <p>NEXUS simulates a five-agent marketplace. Two buyers and two sellers negotiate over resources, with a market manager overseeing stability. Each agent holds a <strong>private configuration</strong> that the others cannot observe:</p>
        <table>
          <thead>
            <tr><th>Agent</th><th>Role</th><th>Budget</th><th>Inventory</th><th>Reservation ₹</th><th>Required qty</th><th>Risk tolerance</th></tr>
          </thead>
          <tbody>
            {Object.entries(AGENT_DESC).map(([id, d]) => (
              <tr key={id}>
                <td><span className={`chip ${d.role === "Purchaser" ? "buyer" : d.role === "Supplier" ? "seller" : "manager"}`}>{id}</span></td>
                <td>{d.role}</td>
                <td>{id === "buyer_a" || id === "buyer_b" ? 10000 : 0}</td>
                <td>{id.startsWith("seller") ? 100 : 0}</td>
                <td>{d.reservation}</td>
                <td>10</td>
                <td>{id === "buyer_a" ? 0.3 : id === "buyer_b" ? 0.6 : id === "seller_a" ? 0.3 : id === "seller_b" ? 0.7 : 0.5}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          Reservation prices are <strong>hidden</strong> from other agents during negotiation — they must infer intent through offers and counteroffers. This information asymmetry is what makes the experiment meaningful (see <code>CONTRACT.md</code> §2 "Observation").
        </p>
        <p style={{ textAlign: "center", marginTop: "0.75rem" }}>
          <Link href="/agent-profiles" style={{ textDecoration: "none", color: "var(--role-buyer-border)", fontWeight: 600 }}>Try it live → Agent Profiles</Link>
        </p>
      </section>

      {/* Section 2: Observe-act-reward loop */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>2. The observe-act-reward loop</h2>
        <p>Each turn, the acting agent observes its local state <code>s_i = [B, I, P, Q, H, M]</code> (budget, inventory, current offer, required quantity, negotiation history, market context), all min-max normalised to [0,1]. It then selects an action:</p>
        <table>
          <thead>
            <tr><th>Head</th><th>Values</th><th>Meaning</th></tr>
          </thead>
            <tbody>
            <tr><td>Action type</td><td>6 (0–5)</td><td>OFFER, COUNTEROFFER, ACCEPT, REJECT, WAIT, WITHDRAW</td></tr>
            <tr><td>Price band</td><td>16 (1–16)</td><td>Price = ₹50 + (band − 1) × ₹10 → ₹50–₹200</td></tr>
            <tr><td>Quantity</td><td>10 (1–10)</td><td>Units 1–10, capped by budget/inventory</td></tr>
          </tbody>
        </table>
        <p>
          The environment validates the action (budget, inventory, reservation constraints). Invalid actions are masked and earn a −0.1 penalty. Valid actions update the shared <code>active_offer</code> slot; ACCEPT by a buyer↔seller pair settles a deal and records it in <code>state.deals</code>. Rewards are agent-specific and shaped (per-step activity/stall penalties plus terminal completion components) — they are RL reward signals, not rupee surplus.
        </p>
        <p style={{ textAlign: "center", marginTop: "0.75rem" }}>
          <Link href="/marketplace" style={{ textDecoration: "none", color: "var(--role-buyer-border)", fontWeight: 600 }}>Try it live → Marketplace</Link>
        </p>
      </section>

      {/* Section 3: MAPPO vs IPPO */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>3. MAPPO vs IPPO — centralized training, decentralized execution</h2>
        <p>NEXUS uses PyTorch MAPPO with CTDE (Centralized Training, Decentralized Execution):</p>
        <ul>
          <li><strong>Training (centralized critic):</strong> A single critic MLP(512, 256, 128) sees the global state <code>S</code> (all 5 local observations concatenated, 254 dims) and outputs a scalar value.</li>
          <li><strong>Execution (decentralized actors):</strong> Each agent's actor MLP(256, 256) sees only its own local observation <code>s_i</code> (50 dims) and outputs 3 categorical heads (type 6, band 16, qty 10). No agent ever sees another's private config at execution time.</li>
          <li>PPO clipping with entropy bonus, GAE advantage estimation, and checkpoints every 100 episodes.</li>
        </ul>
        <p>
          IPPO is the ablation baseline: independent PPO actors with no shared critic. In the measured 5,000-episode run, IPPO collapsed to zero deals — exactly the failure mode the trajectory store exists to diagnose.
        </p>
        <p style={{ textAlign: "center", marginTop: "0.75rem" }}>
          <Link href="/experiment-lab" style={{ textDecoration: "none", color: "var(--role-buyer-border)", fontWeight: 600 }}>Try it live → Experiment Lab</Link>
        </p>
      </section>

      {/* Section 4: NLP layer */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>4. The NLP layer — intent and slot extraction</h2>
        <p>The RL policy decides the action; NLP only communicates it. A DistilBERT model classifies incoming dialogue into one of 6 intent classes (OFFER, COUNTEROFFER, ACCEPT, REJECT, WAIT, WITHDRAW) and extracts slots (price, quantity, resource_type, deadline). The LLM never independently decides negotiation actions.</p>
        <ul>
          <li>Fine-tuned on a synthetic negotiation corpus (3,397 train, 203 test), split by template — not randomly.</li>
          <li>100% accuracy on held-out templates never seen during training; 21 ms per utterance on CPU.</li>
          <li>Round-trip: structured action → natural-language utterance → intent + slots → back to (type, band, qty) via the inverse price map.</li>
        </ul>
        <p style={{ textAlign: "center", marginTop: "0.75rem" }}>
          <Link href="/explainability" style={{ textDecoration: "none", color: "var(--role-buyer-border)", fontWeight: 600 }}>Try it live → Explainability</Link>
        </p>
      </section>

      {/* Section 5: XAI engine */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>5. The XAI engine — trajectory evidence, no generated numbers</h2>
        <p>Every step records a <code>Transition</code>: episode_id, timestep, agent_id, observation, global_state, action, action_log_prob, action_probs, available_actions, reward, reward_components, next_observation, done, utterance, intent, slots, checkpoint_id.</p>
        <ul>
          <li><strong>Reward attribution:</strong> decompose the step reward into components (price_efficiency, speed_bonus, completion_bonus, margin, volume, stability, activity_bonus, stall_penalty, withdraw_penalty, invalid_penalty, constraint_penalty).</li>
          <li><strong>State-action analysis:</strong> correlate observation features with the chosen action.</li>
          <li><strong>Policy analysis:</strong> compare action distributions across checkpoints (KL divergence per head: type, price, quantity).</li>
          <li><strong>Counterfactuals:</strong> "what if the agent had chosen action X instead?" — requires per-step snapshots (not yet exposed via the live API).</li>
        </ul>
        <p>
          All explanations are grounded in recorded trajectory values. When you ask "why?" in the dashboard, the answer cites step numbers, prices, and reservation values from the trace — never LLM-generated speculation.
        </p>
        <p style={{ textAlign: "center", marginTop: "0.75rem" }}>
          <Link href="/explainability" style={{ textDecoration: "none", color: "var(--role-buyer-border)", fontWeight: 600 }}>Try it live → Explainability</Link>
        </p>
      </section>

      {/* Section 6: Interactive deal-zone demo */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>6. Interactive deal-zone illustration</h2>
        <p>
          Adjust buyer and seller reservation prices. A deal zone exists when the buyer's reservation ≥ the seller's reservation. The shaded width shows how much room there is to agree on a price. <strong>This is a client-side illustration using sliders only — no backend data is involved.</strong>
        </p>
        <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", marginBottom: "0.75rem" }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.25rem" }}>Buyer reservation ₹</label>
            <input type="range" min={50} max={200} step={10} value={buyerRes} onChange={(e) => setBuyerRes(Number(e.target.value))} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.2rem" }}>
              <span>₹50</span><span>{buyerRes}</span><span>₹200</span>
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.25rem" }}>Seller reservation ₹</label>
            <input type="range" min={50} max={200} step={10} value={sellerRes} onChange={(e) => setSellerRes(Number(e.target.value))} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.2rem" }}>
              <span>₹50</span><span>{sellerRes}</span><span>₹200</span>
            </div>
          </div>
        </div>
        <div style={{ height: 24, background: "var(--line)", borderRadius: "4px", position: "relative", overflow: "hidden", marginBottom: "0.5rem" }}>
          {dealZone ? (
            <div style={{ position: "absolute", left: `${((sellerRes - 50) / 150) * 100}%`, width: `${(zoneWidth / 150) * 100}%`, height: "100%", background: "var(--role-buyer-border)" }} />
          ) : null}
        </div>
        <p style={{ margin: 0, fontSize: "0.9rem" }}>
          {dealZone ? (
            <>
              <strong>Deal zone exists.</strong> A price between ₹{sellerRes} and ₹{buyerRes} would satisfy both. Width: ₹{zoneWidth}.
            </>
          ) : (
            <>
              <strong>No deal zone.</strong> Buyer's reservation (₹{buyerRes}) is below seller's (₹{sellerRes}); no mutually acceptable price exists in this configuration.
            </>
          )}
        </p>
        <p style={{ fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.5rem" }}>
          Reservation defaults come from <code>src/nexus/env/types.py</code> <code>default_agent_configs()</code> (buyer_a ₹85, seller_a ₹90, etc.).
        </p>
      </section>

      {/* Glossary */}
      <section className="panel" data-anim="scroll-reveal">
        <h2>Glossary</h2>
        <table>
          <tbody>
            <tr><td style={{ fontWeight: 600, width: "140px" }}>AEC</td><td>Agent Environment Cycle — PettingZoo turn-based API where one agent acts at a time.</td></tr>
            <tr><td style={{ fontWeight: 600 }}>CTDE</td><td>Centralized Training, Decentralized Execution — critic sees global state during training; actors see only local obs at execution.</td></tr>
            <tr><td style={{ fontWeight: 600 }}>MAPPO</td><td>Multi-Agent Proximal Policy Optimization — the RL algorithm used to train NEXUS agents.</td></tr>
            <tr><td style={{ fontWeight: 600 }}>IPPO</td><td>Independent PPO — each agent learns its own policy with no shared critic (ablation baseline).</td></tr>
            <tr><td style={{ fontWeight: 600 }}>Transition</td><td>The per-step record stored for explainability: obs, action, reward, components, etc.</td></tr>
            <tr><td style={{ fontWeight: 600 }}>Reward shaping</td><td>Small per-step bonuses/penalties (activity, stall) plus terminal outcome rewards.</td></tr>
            <tr><td style={{ fontWeight: 600 }}>Active offer</td><td>The single (offerer, price, quantity) slot currently on the table; only one offer exists at a time.</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
