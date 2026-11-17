// ============================================================================
// NEXUS E2E Fixtures
//
// Mock responses derived from REAL recorded responses in the backend source
// (src/nexus/api/main.py) and results/*.json. These live ONLY under e2e/.
// ============================================================================

const ACTION_NAMES = ["OFFER", "COUNTEROFFER", "ACCEPT", "REJECT", "WAIT", "WITHDRAW"];

// A realistic step response recorded from a live run (seed 42).
// Per quirk #1: price/qty come from state.active_offer, NOT decoded from action array.
const stepResponse = {
  done: false,
  agent: "buyer_a",
  action: [0, 3, 6],
  reward: 0.01,
  state: {
    episode_id: "ep_0001",
    step: 1,
    agents: ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"],
    agent_selection: "buyer_b",
    active_offer: { offerer: "buyer_a", price: 80, quantity: 6 },
    deals: [],
    budgets: { buyer_a: 10000, buyer_b: 10000, seller_a: 0, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 0, buyer_b: 0, seller_a: 100, seller_b: 100, market_manager: 0 },
  },
};

// Step response where a deal was completed (ACCEPT).
const stepResponseWithDeal = {
  done: false,
  agent: "seller_a",
  action: [2, 3, 6],
  reward: 0.0,
  state: {
    episode_id: "ep_0001",
    step: 4,
    agents: ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"],
    agent_selection: "buyer_b",
    active_offer: null,
    deals: [{ buyer: "buyer_a", seller: "seller_a", price: 80, quantity: 6, timestep: 4 }],
    budgets: { buyer_a: 9520, buyer_b: 10000, seller_a: 480, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 6, buyer_b: 0, seller_a: 94, seller_b: 100, market_manager: 0 },
  },
};

// Done=true response (all agents done or max_steps reached) - no agent field per quirk #2.
const stepResponseDone = {
  done: true,
  state: {
    episode_id: "ep_0001",
    step: 100,
    agents: [],
    agent_selection: null,
    active_offer: null,
    deals: [{ buyer: "buyer_a", seller: "seller_a", price: 80, quantity: 6, timestep: 4 }],
    budgets: { buyer_a: 9520, buyer_b: 10000, seller_a: 480, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 6, buyer_b: 0, seller_a: 94, seller_b: 100, market_manager: 0 },
  },
};

// Start response.
const startResponse = {
  episode_id: "ep_0001",
  state: {
    episode_id: "ep_0001",
    step: 0,
    agents: ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"],
    agent_selection: "buyer_a",
    active_offer: null,
    deals: [],
    budgets: { buyer_a: 10000, buyer_b: 10000, seller_a: 0, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 0, buyer_b: 0, seller_a: 100, seller_b: 100, market_manager: 0 },
  },
};

// WAIT state: no message, waiting marker only.
const stepResponseWait = {
  done: false,
  agent: "market_manager",
  action: [4, 1, 1],
  reward: -0.01,
  state: {
    episode_id: "ep_0001",
    step: 2,
    agents: ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"],
    agent_selection: "buyer_a",
    active_offer: { offerer: "buyer_a", price: 80, quantity: 6 },
    deals: [],
    budgets: { buyer_a: 10000, buyer_b: 10000, seller_a: 0, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 0, buyer_b: 0, seller_a: 100, seller_b: 100, market_manager: 0 },
  },
};

// WITHDRAW state: offer retracts.
const stepResponseWithdraw = {
  done: false,
  agent: "buyer_b",
  action: [5, 1, 1],
  reward: -0.1,
  state: {
    episode_id: "ep_0001",
    step: 3,
    agents: ["buyer_a", "seller_a", "seller_b", "market_manager"],
    agent_selection: "seller_a",
    active_offer: null,
    deals: [],
    budgets: { buyer_a: 10000, buyer_b: 10000, seller_a: 0, seller_b: 0, market_manager: 0 },
    inventories: { buyer_a: 0, buyer_b: 0, seller_a: 100, seller_b: 100, market_manager: 0 },
  },
};

// Query response with full evidence.
const queryResponse = {
  answer:
    "Based on recorded trajectory ep_0001 step 0: buyer_a chose action [0, 3, 6] with reward 0.010. Its configured reservation price is ₹85. All numbers come from the trajectory/config, not generation.",
  evidence: [
    {
      type: "reservation_price",
      field: "reservation_price",
      value: 85.0,
      source: "config",
      episode_id: "ep_0001",
      agent_id: "buyer_a",
    },
    {
      type: "action",
      field: "action",
      value: [0, 3, 6],
      source: "trajectory",
      episode_id: "ep_0001",
      timestep: 0,
      agent_id: "buyer_a",
    },
    {
      type: "reward_component",
      field: "reward",
      value: 0.01,
      source: "trajectory",
      episode_id: "ep_0001",
      timestep: 0,
      agent_id: "buyer_a",
    },
  ],
  confidence: 0.9,
  sources: ["ep_0001"],
};

// Query response with thin evidence (single item).
const queryResponseThin = {
  answer: "Recorded step 0 in ep_0001: agent buyer_a chose action [0, 3, 6] with reward 0.010.",
  evidence: [
    {
      type: "action",
      field: "action",
      value: [0, 3, 6],
      source: "trajectory",
      episode_id: "ep_0001",
      timestep: 0,
      agent_id: "buyer_a",
    },
  ],
  confidence: 0.8,
  sources: ["ep_0001"],
};

// Query response with unknown evidence type (generic fallback).
const queryResponseUnknown = {
  answer: "No specific explanation available.",
  evidence: [
    {
      type: "unknown_type",
      field: "custom_field",
      value: { some: "data" },
      source: "trajectory",
      episode_id: "ep_0001",
      agent_id: "buyer_a",
      timestep: 0,
    },
  ],
  confidence: 0.5,
  sources: ["ep_0001"],
};

// Trajectory response (for replay).
const trajectoryResponse = [
  {
    episode_id: "ep_0001",
    timestep: 0,
    agent_id: "buyer_a",
    action: [0, 3, 6],
    reward: 0.01,
    reward_components: { activity_bonus: 0.01 },
    utterance: "",
    intent: "",
    slots: {},
  },
  {
    episode_id: "ep_0001",
    timestep: 1,
    agent_id: "seller_a",
    action: [1, 5, 4],
    reward: 0.01,
    reward_components: { activity_bonus: 0.01 },
    utterance: "",
    intent: "",
    slots: {},
  },
  {
    episode_id: "ep_0001",
    timestep: 2,
    agent_id: "buyer_b",
    action: [2, 3, 6],
    reward: 0.0,
    reward_components: { completion_bonus: 0.0 },
    utterance: "",
    intent: "",
    slots: {},
  },
];

// Agents response (for Agent Profiles / research view).
const agentsResponse = [
  { agent_id: "buyer_a", role: "buyer", budget: 10000, inventory: 0, reservation_price: 85.0, required_qty: 10, risk_tolerance: 0.3 },
  { agent_id: "buyer_b", role: "buyer", budget: 10000, inventory: 0, reservation_price: 110.0, required_qty: 10, risk_tolerance: 0.6 },
  { agent_id: "seller_a", role: "seller", budget: 0.0, inventory: 100, reservation_price: 90.0, required_qty: 10, risk_tolerance: 0.3, unit_cost: 60.0 },
  { agent_id: "seller_b", role: "seller", budget: 0.0, inventory: 100, reservation_price: 80.0, required_qty: 10, risk_tolerance: 0.7, unit_cost: 55.0 },
  { agent_id: "market_manager", role: "manager", budget: 0.0, inventory: 0, reservation_price: 0.0, required_qty: 0, risk_tolerance: 0.5 },
];

// summary.json shape (from results/summary.json).
const summaryJson = {
  mode: "full",
  train_episodes_per_algo: 5000,
  eval_episodes: 100,
  exp1: {
    results: {
      mappo: {
        mean_reward: 0.09211428571428573,
        cumulative_reward_per_episode: 0.4605714285714285,
        transaction_rate: 1.0,
        negotiation_duration_steps: 40.0,
        buyer_savings: 0.44155844155844154,
        seller_profit: 0.023809523809523832,
        deals_total: 700,
        episodes: 100,
      },
      ippo: {
        mean_reward: -0.0059999999999999975,
        cumulative_reward_per_episode: -0.029999999999999968,
        transaction_rate: 0.0,
        negotiation_duration_steps: 40.0,
        buyer_savings: 0.0,
        seller_profit: 0.0,
        deals_total: 0,
        episodes: 100,
      },
      rule_based: {
        mean_reward: 0.06021807486631017,
        cumulative_reward_per_episode: 0.3010903743315508,
        transaction_rate: 0.56,
        negotiation_duration_steps: 35.24,
        buyer_savings: 0.11397461503769091,
        seller_profit: 0.5976633807959107,
        deals_total: 83,
        episodes: 100,
      },
      random: {
        mean_reward: -0.35692464451235045,
        cumulative_reward_per_episode: -1.7846232225617522,
        transaction_rate: 0.46,
        negotiation_duration_steps: 28.58,
        buyer_savings: -0.31488413547235045,
        seller_profit: 1.1747474747474747,
        deals_total: 60,
        episodes: 100,
      },
    },
    mappo_wins: ["mean_reward", "cumulative_reward_per_episode", "transaction_rate", "buyer_savings"],
    mappo_wins_count: 4,
    success_criterion: "MAPPO best on >=3 of the 5 PRD metrics (reward, savings, profit, transaction rate, duration)",
  },
  ippo_500_gate: {
    first50_mean_reward: -0.03581193582887702,
    episodes_451_500_mean_reward: 0.1521552121212121,
    improved: true,
  },
  exp4_policy_shift: {
    checkpoints: [100, 5000],
    mean_kl: 0.6645066661928393,
    kl_by_head_type_price_qty: [1.0561369574135653, 0.46364065145070976, 0.4737423897142429],
  },
};

// bert_metrics_full.json shape.
const bertMetrics = {
  model: "distilbert-base-uncased",
  train_n: 3397,
  test_n: 203,
  held_out_templates: ["ACCEPT_x7", "COUNTEROFFER_x7", "OFFER_x7", "REJECT_x7", "WAIT_x7", "WITHDRAW_x7"],
  accuracy: 1.0,
  epochs: 3,
  device: "cpu",
  train_seconds: 705.0459570884705,
  batch_latency_s_median: 1.1060220500003197,
};

// ============================================================================
// Route mock helpers
// ============================================================================

/**
 * Mock a successful /api/v1/simulation/start + first /step.
 */
export function mockStartAndStep(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(startResponse),
  });
}

/**
 * Mock /step returning stepResponse.
 */
export function mockStep(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(stepResponse),
  });
}

/**
 * Mock /step returning stepResponseWithDeal.
 */
export function mockStepWithDeal(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(stepResponseWithDeal),
  });
}

/**
 * Mock /step returning done=true.
 */
export function mockStepDone(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(stepResponseDone),
  });
}

/**
 * Mock /step returning WAIT.
 */
export function mockStepWait(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(stepResponseWait),
  });
}

/**
 * Mock /step returning WITHDRAW.
 */
export function mockStepWithdraw(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(stepResponseWithdraw),
  });
}

/**
 * Mock /query with full evidence.
 */
export function mockQuery(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(queryResponse),
  });
}

/**
 * Mock /query with thin evidence.
 */
export function mockQueryThin(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(queryResponseThin),
  });
}

/**
 * Mock /query with unknown evidence type.
 */
export function mockQueryUnknown(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(queryResponseUnknown),
  });
}

/**
 * Mock /query returning 404 (no episode).
 */
export function mockQueryNotFound(route) {
  route.fulfill({
    status: 404,
    contentType: "application/json",
    body: JSON.stringify({ detail: "no episode; start a simulation first" }),
  });
}

/**
 * Mock /agents.
 */
export function mockAgents(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(agentsResponse),
  });
}

/**
 * Mock /episodes/{id}/trajectory.
 */
export function mockTrajectory(route) {
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(trajectoryResponse),
  });
}

/**
 * Mock /health failing.
 */
export function mockHealthFail(route) {
  route.fulfill({
    status: 500,
    contentType: "application/json",
    body: JSON.stringify({ detail: "internal error" }),
  });
}

/**
 * Mock /step failing (500).
 */
export function mockStepFail(route) {
  route.fulfill({
    status: 500,
    contentType: "application/json",
    body: JSON.stringify({ detail: "step error" }),
  });
}

/**
 * Mock /step that never responds (timeout).
 */
export function mockStepTimeout(route) {
  // Never fulfill - simulates a hanging request.
}

/**
 * Mock summary.json missing (Experiment Lab empty state).
 */
export function mockSummaryMissing(route) {
  route.fulfill({
    status: 404,
    contentType: "application/json",
    body: JSON.stringify({ detail: "not found" }),
  });
}

export {
  stepResponse,
  startResponse,
  queryResponse,
  queryResponseThin,
  queryResponseUnknown,
  summaryJson,
  bertMetrics,
  agentsResponse,
  trajectoryResponse,
};
