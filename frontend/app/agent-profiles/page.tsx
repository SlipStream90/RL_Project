"use client";
import { useEffect, useState } from "react";
import { api, type AgentConfig } from "../../lib/api";

export default function Agents() {
  const [rows, setRows] = useState<AgentConfig[]>([]);
  const [err, setErr] = useState("");
  const [highlight, setHighlight] = useState<string | null>(null);

  useEffect(() => {
    // Read ?agent= query param for highlight from network-graph click
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const a = params.get("agent");
      if (a && AGENT_IDS.includes(a as any)) setHighlight(a);
    }
    api("/api/v1/agents")
      .then((data) => setRows(data as AgentConfig[]))
      .catch((e) => setErr(e.message));
  }, []);

  if (err) {
    return (
      <div>
        <h1>Agent Profiles</h1>
        <div className="alert" role="alert">{err}. Backend offline?</div>
      </div>
    );
  }

  return (
    <div>
      <h1>Agent Profiles</h1>
      <p className="lede">Private configuration per agent. Reservation prices stay hidden from other agents inside the environment; they are shown here for research inspection.</p>
      <div className="panel">
        <table aria-label="Agents">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Role</th>
              <th>Budget</th>
              <th>Inventory</th>
              <th>Reservation ₹</th>
              <th>Needs</th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => {
              const isHighlight = highlight === a.agent_id;
              return (
                <tr key={a.agent_id} style={{ background: isHighlight ? "rgba(15, 107, 79, 0.12)" : "inherit" }}>
                  <td><span className={`chip ${a.role}`}>{a.agent_id}</span></td>
                  <td>{a.role}</td>
                  <td>{a.budget}</td>
                  <td>{a.inventory}</td>
                  <td>{a.reservation_price}</td>
                  <td>{a.required_qty}</td>
                  <td>{a.risk_tolerance}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length && <p className="empty">No agent data available.</p>}
      </div>
    </div>
  );
}

const AGENT_IDS: string[] = ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"];
