export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const ACTION_NAMES = ["OFFER", "COUNTEROFFER", "ACCEPT", "REJECT", "WAIT", "WITHDRAW"];

// Price band math mirrors src/nexus/env/types.py: price = 50 + (band-1)*10
export const P_MIN = 50;
export const P_STEP = 10;

export function bandToPrice(band: number | null | undefined): number | null {
  if (band == null) return null;
  return P_MIN + (Number(band) - 1) * P_STEP;
}

export function actionLabel(typeIndex: number | null | undefined): string {
  if (typeIndex == null) return "—";
  const n = Number(typeIndex);
  if (n >= 0 && n < ACTION_NAMES.length) return ACTION_NAMES[n];
  return String(n);
}

export function agentRole(agentId: string): "buyer" | "seller" | "manager" | "unknown" {
  if (agentId.startsWith("buyer")) return "buyer";
  if (agentId.startsWith("seller")) return "seller";
  if (agentId === "market_manager") return "manager";
  return "unknown";
}

export async function api(path: string, init?: RequestInit) {
  const r = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

export type SimState = {
  episode_id: string;
  step: number;
  agents: string[];
  agent_selection: string | null;
  active_offer: { offerer: string; price: number; quantity: number } | null;
  deals: unknown[];
  budgets: Record<string, number>;
  inventories: Record<string, number>;
};

export type StepResult = {
  done: boolean;
  agent: string;
  action: number[];
  reward: number;
  state: SimState;
};

export type TrajectoryStep = {
  episode_id: string;
  timestep: number;
  agent_id: string;
  action: number[];
  reward: number;
  reward_components: Record<string, number>;
  utterance: string;
  intent: string;
  slots: Record<string, unknown>;
};

export type QueryResult = {
  answer: string;
  evidence: Array<{
    type: string;
    field: string;
    value: unknown;
    source: string;
    episode_id?: string;
    agent_id?: string;
    timestep?: number;
  }>;
  confidence: number;
  sources: string[];
};

export type AgentConfig = {
  agent_id: string;
  role: string;
  budget: number;
  inventory: number;
  reservation_price: number;
  required_qty: number;
  risk_tolerance: number;
};
