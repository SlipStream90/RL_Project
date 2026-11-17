# NEXUS — Frozen Integration Contract v1

Single source of truth. All weeks build against this file. Changes require a contract bump, not silent drift.

## Env
- Class `NexusMarketplaceEnv(AECEnv)`, `metadata={"render_modes":["human"],"name":"nexus_marketplace_v0"}`.
- `possible_agents = ["buyer_a","buyer_b","seller_a","seller_b","market_manager"]` (string IDs everywhere).
- Lifecycle: `reset(seed=None, options=None)`, `step(action)`, `observe(agent)`, `last()`, `agent_iter()`, `state()`, `observation_space(agent)`, `action_space(agent)`, `render()`, `close()`.
- Termination uses `terminations` + `truncations` (never `dones`). `step(None)` on a dead agent is legal. `done(a) = terminations[a] or truncations[a]`.
- Fixed round-robin turn order over live agents. Single `active_offer=(offerer, price, quantity)`; any non-offerer buyer/seller may respond. `market_manager` may only WAIT/WITHDRAW.
- Snapshot: `get_state()` / `set_state(state)` deep-copy round-trip; `get_global_state()` returns critic vector.

## Observation (frozen dims)
- `s_i = [B, I, P, Q, H, M]`, all min-max normalised to [0,1], `float32`.
- H (k=5): 5 offer prices + 5 quantities + 5x6 intent one-hot + 1 time-since-last-offer = 41.
- M: 5 signals (avg tx price, n active buyers, n active sellers, time remaining, scarcity).
- `d_s = 4 + 41 + 5 = 50`. Global `S = concat(s_1..s_5, 4 aggregates)` = 254. Critic input = global state only.

## Action
- `MultiDiscrete([6, 16, 10])` = (type, price_band, quantity). Types: 0 OFFER, 1 COUNTEROFFER, 2 ACCEPT, 3 REJECT, 4 WAIT, 5 WITHDRAW.
- Price: `price = 50 + (band-1)*10`, bands 1..16 (₹50–₹200). Inverse: `band = clip(round((price-50)/10)+1, 1, 16)`.
- Quantity 1..10 (`Q_max=10`). Per-head masks via `action_masks(agent)` returning dict of 3 boolean arrays. Log-prob = sum of head log-probs. `action_probs` stored as 3 head distributions.
- Invalid semantics (one only): masked actions are never sampled; a directly forced invalid `step()` gives reward -0.1, no state change (except turn advance), `info["invalid"]=True`.

## Rewards
- Env emits per-step `reward_components: dict[str,float]` that sums to the step reward. Terminal components use the same keys. Keys: `price_efficiency, speed_bonus, completion_bonus, margin, volume, stability, activity_bonus, stall_penalty, withdraw_penalty, invalid_penalty, constraint_penalty`.
- Role weights per NEXUS_ENV_SPEC §6 (Buyer A price-focused, Buyer B speed-focused, Seller A margin, Seller B volume, Manager stability).

## Trajectory (single schema, Week 1 onward)
`Transition` dataclass in `src/nexus/env/types.py`: `episode_id, timestep, agent_id, observation, global_state, action, action_log_prob, action_probs, available_actions, reward, reward_components, next_observation, done, utterance, intent, slots, checkpoint_id`.
Recorder ships in Week 1 (`env.get_trajectory(episode_id)`). Later weeks read it; they never recompute rewards.

## RL (Week 2)
- IPPO before MAPPO. Actor MLP(256,256) -> 3 heads (6,16,10). Centralised critic MLP(512,256,128) -> scalar, input 254 only.
- Checkpoints: weights + `config_hash` + norm stats + `episode_count`. CTDE: inference uses `observe()` only.

## NLP (Week 3)
- RL decides, NLP communicates. Intents = 6 action types. Slots: price, quantity, resource_type, deadline (deadline/resource have no env field; carried in dialogue only).
- Round-trip must recover (type, band, qty) via the inverse price map above. Corpus split by template, not randomly.

## API (Week 5)
- DTOs derive from `Transition`. Canonical paths per PRD_Week5 §4.1 (`/api/v1/...`). Explanations: LLM formats pre-extracted evidence only; every number must exist in a trajectory fixture.

## Verification lanes
- Parallel: masking/reward/baseline units, checkpoint IO, NLP round-trip, XAI units, TestClient routes, per-section Playwright.
- Main-thread gates: PettingZoo `api_test`, seed determinism, CTDE leak check, evidence grounding, WebSocket live flow, coverage (env 80 / rl 70 / xai 75 / api 70).
- Smokes never prove expensive gates (500-ep IPPO, BERT >=85%, 5000-ep Exp 1). Full runs are nightly/release only.
