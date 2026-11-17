import { readFile, readdir } from "fs/promises";
import { join } from "path";

const ALGOS = ["mappo", "ippo", "rule_based", "random"];
const ALGO_LABELS = { mappo: "MAPPO", ippo: "IPPO", rule_based: "Rule-based", random: "Random" };

const METRICS = [
  { key: "mean_reward", label: "Mean reward", higherIsBetter: true, format: (v) => v.toFixed(3) },
  { key: "buyer_savings", label: "Buyer savings", higherIsBetter: true, format: (v) => v.toFixed(3) },
  { key: "transaction_rate", label: "Transaction rate", higherIsBetter: true, format: (v) => (v * 100).toFixed(1) + "%" },
  { key: "seller_profit", label: "Seller profit", higherIsBetter: true, format: (v) => v.toFixed(3) },
  { key: "deals_total", label: "Deals completed", higherIsBetter: true, format: (v) => Math.round(v).toString() },
  { key: "negotiation_duration_steps", label: "Duration (steps)", higherIsBetter: false, format: (v) => v.toFixed(1) },
];

async function readJson(paths) {
  for (const p of paths) {
    try {
      const raw = await readFile(p, "utf-8");
      return JSON.parse(raw);
    } catch { /* try next */ }
  }
  return null;
}

async function listCheckpoints() {
  const dirs = [join(process.cwd(), "..", "results", "checkpoints"), join(process.cwd(), "results", "checkpoints")];
  for (const dir of dirs) {
    try {
      const files = await readdir(dir);
      return files.filter((f) => f.endsWith(".pt")).sort();
    } catch { /* try next */ }
  }
  return [];
}

function roleColor(algo) {
  switch (algo) {
    case "mappo": return "var(--role-buyer)";
    case "rule_based": return "var(--role-seller)";
    case "ippo": return "var(--role-manager)";
    default: return "var(--ink)";
  }
}

// Async Server Component — Next.js awaits the returned promise.
export default async function ExperimentLab() {
  const summary = await readJson([
    join(process.cwd(), "..", "results", "summary.json"),
    join(process.cwd(), "results", "summary.json"),
  ]);
  const bert = await readJson([
    join(process.cwd(), "..", "results", "bert_metrics_full.json"),
    join(process.cwd(), "results", "bert_metrics_full.json"),
  ]);
  const checkpoints = await listCheckpoints();

  if (!summary) {
    return (
      <div className="dashboard-page" data-anim="page-transition">
        <h1>Experiment Lab</h1>
        <p className="lede">Experiment results are not available.</p>
        <div className="panel">
          <p className="empty"><code>results/summary.json</code> not found. Run the Week 6 harness to produce it.</p>
        </div>
      </div>
    );
  }

  const results = summary.exp1.results;
  const mappoWins = summary.exp1.mappo_wins;
  const hasData = ALGOS.some((a) => results[a] != null);

  return (
    <div className="dashboard-page" data-anim="page-transition">
      <h1>Experiment Lab</h1>
      <p className="lede">
        Experiment 1 results read from <code>results/summary.json</code> (Week 6 harness output).
        {summary.train_episodes_per_algo} training episodes per algorithm, {summary.eval_episodes} evaluation episodes.
        No numbers are fabricated — every value comes from the harness JSON.
      </p>

      {hasData ? (
        <>
          <section className="panel">
            <h2>Algorithm comparison</h2>
            <p style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: "1rem" }}>
              Grouped bars per metric. Each group shows all four algorithms side by side. Winner marked per metric.
            </p>
            {METRICS.map((m) => {
              const values = ALGOS.map((a) => ({ algo: a, v: results[a]?.[m.key] }));
              const hasValues = values.some((x) => x.v != null);
              if (!hasValues) return null;
              const maxVal = Math.max(...values.map((x) => x.v ?? 0));
              const isHigherBetter = m.higherIsBetter;
              const best = values.filter((x) => x.v != null).sort((a, b) => (isHigherBetter ? (b.v - a.v) : (a.v - b.v)))[0];
              return (
                <div key={m.key} style={{ marginBottom: "1.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                    <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>{m.label}</span>
                    {best && mappoWins.includes(m.key) && (
                      <span className="chip" style={{ fontSize: "0.7rem", padding: "0.1rem 0.5rem", background: "var(--status-accept-fill)", color: "var(--status-accept)", borderColor: "var(--status-accept)" }}>MAPPO win</span>
                    )}
                    {best && !mappoWins.includes(m.key) && (
                      <span className="chip" style={{ fontSize: "0.7rem", padding: "0.1rem 0.5rem", background: "var(--status-wait-fill)", color: "var(--status-wait)", borderColor: "var(--status-wait)" }}>Not MAPPO</span>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: "1rem", marginBottom: "0.5rem", alignItems: "flex-end" }}>
                    {values.map((x) => {
                      if (x.v == null) return null;
                      const pct = maxVal > 0 ? (x.v / maxVal) * 100 : 0;
                      const isBest = best && x.algo === best.algo;
                      return (
                        <div key={x.algo} style={{ display: "flex", flexDirection: "column", gap: "0.25rem", minWidth: 100 }}>
                          <span style={{ fontSize: "0.75rem", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{ALGO_LABELS[x.algo]}</span>
                          <div style={{ height: 14, background: "var(--line)", borderRadius: "var(--radius-sm)", position: "relative", overflow: "hidden", width: "100%" }}>
                            <div
                              style={{
                                height: "100%",
                                width: `${Math.max(pct, 4)}%`,
                                background: isBest ? "var(--status-accept)" : roleColor(x.algo),
                                borderRadius: "var(--radius-sm)",
                                transformOrigin: "left center",
                                animation: "bar-grow 0.6s ease forwards",
                              }}
                              data-anim="lab-bars"
                            />{isBest ? <span data-testid="winner-badge">winner</span> : null}
                          </div>
                          <span className="mono" style={{ fontSize: "0.8rem" }}>{m.format(x.v)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </section>

          <section className="panel">
            <h2>IPPO 500-episode gate</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
              <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>First 50 mean reward</div>
                <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{summary.ippo_500_gate?.first50_mean_reward?.toFixed(3) ?? "—"}</div>
              </div>
              <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Episodes 451–500 mean reward</div>
                <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{summary.ippo_500_gate?.episodes_451_500_mean_reward?.toFixed(3) ?? "—"}</div>
              </div>
              <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Improved</div>
                <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{summary.ippo_500_gate?.improved ? "Yes" : "No"}</div>
              </div>
            </div>
          </section>

          <section className="panel">
            <h2>Strategy evolution (checkpoints)</h2>
            {checkpoints.length > 0 ? (
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                {checkpoints.map((f) => (
                  <span key={f} className="chip" style={{ fontSize: "0.75rem" }}>{f.replace("_ckpt_", " · ").replace(".pt", "")}</span>
                ))}
              </div>
            ) : (
              <p className="empty">No checkpoint files found in results/checkpoints/. Run Week 6 harness to produce them.</p>
            )}
            {summary.exp4_policy_shift && (
              <div style={{ marginTop: "1rem", padding: "0.75rem", background: "var(--paper)", border: "1px solid var(--line)", borderRadius: "var(--radius-sm)" }}>
                <p style={{ fontSize: "0.82rem", color: "var(--muted)", margin: 0 }}>
                  MAPPO mean policy shift (KL, 100→5000 eps): <span style={{ fontWeight: 600 }}>{summary.exp4_policy_shift.mean_kl.toFixed(3)}</span> (type {summary.exp4_policy_shift.kl_by_head_type_price_qty?.[0]?.toFixed(2)}, price {summary.exp4_policy_shift.kl_by_head_type_price_qty?.[1]?.toFixed(2)}, qty {summary.exp4_policy_shift.kl_by_head_type_price_qty?.[2]?.toFixed(2)})
                </p>
              </div>
            )}
          </section>

          <section className="panel">
            <h2>NLP (DistilBERT intent classification)</h2>
            {bert ? (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem" }}>
                  <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Accuracy (held-out templates)</div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{(bert.accuracy * 100).toFixed(0)}%</div>
                  </div>
                  <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Test utterances</div>
                    <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{bert.test_n}</div>
                  </div>
                  <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Model</div>
                    <div style={{ fontSize: "1rem", fontWeight: 600 }}>{bert.model}</div>
                  </div>
                  <div className="panel" style={{ padding: "1rem", margin: 0 }}>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Device</div>
                    <div style={{ fontSize: "1rem", fontWeight: 600 }}>{bert.device}</div>
                  </div>
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.5rem" }}>
                  Held-out templates: {bert.held_out_templates.join(", ")}. Accuracy measured on templates never seen during training.
                </p>
              </div>
            ) : (
              <p className="empty"><code>results/bert_metrics_full.json</code> not found.</p>
            )}
          </section>

          <div className="notice" style={{ marginTop: "1rem", fontSize: "0.85rem" }}>
            <strong>Success criterion:</strong> {summary.exp1.success_criterion}. MAPPO won {summary.exp1.mappo_wins_count} of 5 metrics: {mappoWins.join(", ")}.
          </div>
        </>
      ) : (
        <div className="panel">
          <p className="empty">No experiment results found in <code>results/summary.json</code>. Run the Week 6 harness to produce measured numbers.</p>
        </div>
      )}
    </div>
  );
}
