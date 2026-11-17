# NEXUS — Negotiating Agents & Explainability System

**Product Requirements Document (PRD)**  
Version: 1.0  
Date: November 2026  
Status: Draft for Review  

---

## 1. Executive Summary

NEXUS is a software-only research platform that simulates a multi-agent marketplace where autonomous AI agents negotiate over resources through natural language. Agents learn negotiation strategies via multi-agent reinforcement learning (MARL), and an explainability engine (XAI) investigates why they make particular decisions.

NEXUS answers two core research questions:

1. **Can autonomous agents learn better negotiation strategies through repeated interactions?**
2. **Can we understand, explain, and diagnose the strategies they develop?**

The platform combines three technical components: multi-agent reinforcement learning, conversational AI, and explainable AI. The primary deliverable is an interactive dashboard that lets researchers replay negotiations, compare algorithms, and ask conversational questions about agent behaviour — with answers grounded in actual simulation data (states, actions, rewards, policy probabilities), not speculative LLM generation.

---

## 2. Problem Statement

Most multi-agent negotiation systems today fall into one of two camps:

- **Rule-based agents** — deterministic, brittle, and unable to adapt to new opponents or market conditions.
- **LLM-driven agents** — fluent in natural language but opaque; it is impossible to tell whether an LLM is negotiating strategically or merely mimicking human dialogue patterns. RL improvement cannot be isolated.

There is a gap for a platform where:

- Agents negotiate in natural language **but** their decisions are driven by learned RL policies (not LLMs).
- Every decision is recorded with full observability (observation, action, action probabilities, reward, state transition).
- Researchers can run controlled experiments comparing MAPPO, independent PPO, and rule-based baselines.
- Stakeholders can interrogate *why* an agent accepted, rejected, or counteroffered, with evidence from actual trajectories.

NEXUS fills this gap.

---

## 3. Goals & Objectives

### Primary Goals

| # | Goal | Success Criterion |
|---|------|-------------------|
| G1 | Build a PettingZoo-compliant multi-agent marketplace environment with 5 agents (2 buyers, 2 sellers, 1 market manager). | Environment runs stable negotiation episodes; all agents observe partial state and act via discrete action space. |
| G2 | Train agents with MAPPO (CTDE) and compare against independent PPO and rule-based baselines. | MAPPO agents achieve higher average reward / buyer savings / seller profit than baselines over 5,000+ episodes. |
| G3 | Connect learned policies to natural-language dialogue via a BERT intent classifier + slot extractor + dialogue generator. | Structured RL actions ↔ natural language conversion with ≥85% intent classification accuracy on a held-out negotiation corpus. |
| G4 | Build an explainability engine that records trajectories and supports reward attribution, state-action analysis, policy evolution, counterfactuals, and exploit detection. | Dashboard can answer ≥5 predefined investigation questions with evidence from recorded data. |
| G5 | Deliver a conversational dashboard (FastAPI + Next.js) for negotiation replay, agent profiles, experiment lab, and natural-language queries. | User can ask "Why did Buyer A reject Seller A's first offer?" and receive an evidence-backed answer in <5 seconds. |

### Secondary Goals

- Reproducible experiment tracking (TensorBoard / MLflow).
- Strategy evolution timeline (compare agent behaviour at 100, 500, 1,000, 5,000 episodes).
- GPU-accelerated policy training and inference.
- All software-only; no hardware dependencies beyond a standard GPU workstation.

---

## 4. Research Questions (Experiments)

NEXUS is structured around four evaluable experiments:

### Experiment 1: Do agents become better negotiators?
Compare random, rule-based, independent PPO, and MAPPO agents. Measure average buyer savings, seller profit, transaction rate, negotiation duration, and cumulative reward over time.

### Experiment 2: Do agents develop different personalities?
Assign distinct reward functions (aggressive buyer, opportunistic seller, cooperative seller, risk-averse buyer). Analyse whether distinguishable negotiation strategies emerge.

### Experiment 3: Can agents exploit each other?
Introduce information asymmetry (e.g., one seller knows true inventory; buyers receive partial market info). Investigate strategic withholding, lowballing, collusion-like behaviour, and exploitation of predictable opponents.

### Experiment 4: Can we explain learned negotiation strategies?
Compare an agent early vs. late in training. Analyse action-probability shifts, observation influence, reward-component drivers, and generalization to unseen opponents.

---

## 5. Target Users & Personas

| Persona | Role | Needs |
|---------|------|-------|
| RL Researcher | Academic / industry researcher studying MARL | Reproducible training runs, experiment tracking, baseline comparisons, policy export. |
| XAI Researcher | Researcher studying explainable AI | Trajectory logs, counterfactual rollouts, reward attribution, policy diagnostics. |
| Product Demo User | Stakeholder / instructor | Live negotiation replay, conversational Q&A, dashboard visualizations. |
| ML Engineer | Platform maintainer | FastAPI backend, Next.js frontend, clear API contracts, Docker deployability. |

---

## 6. Functional Requirements

### 6.1 Marketplace Simulation (Subsystem A)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | The environment must implement the PettingZoo AEC / Parallel API so it is interchangeable with other MARL environments. | Must |
| FR-2 | Five agents: Buyer A, Buyer B, Seller A, Seller B, Market Manager. | Must |
| FR-3 | Each agent has private configuration: starting capital, inventory, reservation price, resource requirements, risk tolerance, private information. | Must |
| FR-4 | Agent state: sᵢ = [Bᵢ, Iᵢ, Pᵢ, Qᵢ, Hᵢ] (budget, inventory, current offer, required quantity, negotiation-history features). | Must |
| FR-5 | Discrete action space: {Offer, Counteroffer, Accept, Reject, Wait, Withdraw}. Price bands (discrete pricing) to start; continuous pricing as a future extension. | Must |
| FR-6 | Negotiation mechanics: offer → counteroffer → accept/reject → transaction completion or withdrawal. | Must |
| FR-7 | Rule-based baseline agents (fixed strategy) for comparison. | Must |
| FR-8 | Configurable episode length and market conditions (resource scarcity, demand shocks). | Should |

### 6.2 RL Engine (Subsystem B)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-9 | Implement MAPPO with CTDE: centralized critic (global state) during training, decentralized actors (local observations) during execution. | Must |
| FR-10 | Implement Independent PPO baseline for ablation. | Must |
| FR-11 | Agent-specific reward functions aligned with each agent's objective (e.g., buyer savings, seller profit, transaction completion, market stability). | Must |
| FR-12 | PyTorch policy networks (actor + critic). | Must |
| FR-13 | Gymnasium integration for environment conventions. | Must |
| FR-14 | Experiment tracking via TensorBoard / MLflow (reward curves, loss, entropy, KL divergence). | Must |
| FR-15 | GPU support for training and inference (CUDA). | Should |
| FR-16 | Policy checkpointing at configurable intervals (e.g., every 100 episodes). | Must |

### 6.3 Conversational Negotiation Engine (Subsystem C)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-17 | Structured RL action → natural language dialogue generation (template-based or fine-tuned LLM). | Must |
| FR-18 | Incoming natural language → intent classification + slot extraction (offer price, quantity, deadline). | Must |
| FR-19 | BERT fine-tuned for intent classes: OFFER, COUNTEROFFER, ACCEPT, REJECT, WITHDRAW, WAIT. | Must |
| FR-20 | Slot extraction for price, quantity, and resource type from dialogue. | Must |
| FR-21 | Offer validation against environment constraints (budget, inventory, reservation price). | Must |
| FR-22 | The RL policy chooses the action; NLP only provides the communication interface. LLMs must not independently decide negotiation actions. | Must |
| FR-23 | Multi-turn dialogue state tracking across a negotiation episode. | Should |

### 6.4 Explainability Engine (Subsystem D)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-24 | Record per-step trajectory: observation, chosen action, action probabilities, available alternatives, negotiation history, reward, reward components, state transition. | Must |
| FR-25 | Store policy checkpoints for temporal comparison. | Must |
| FR-26 | Reward attribution: decompose reward into components per decision. | Must |
| FR-27 | State-action analysis: identify which observations drive which decisions. | Must |
| FR-28 | Policy analysis: compare action distributions across training checkpoints. | Must |
| FR-29 | Counterfactual analysis: simulate "what if the agent had chosen action X instead of Y?" using recorded trajectories. | Must |
| FR-30 | Exploit detection: flag strategies that systematically advantage one agent at another's expense (e.g., strategic withholding, repeated lowballing). | Should |
| FR-31 | Generalization testing: evaluate trained agents against unseen opponent configurations. | Should |
| FR-32 | SHAP / Captum feature attribution on a surrogate model or value function, complemented by policy probabilities and counterfactual rollouts (SHAP alone is not sufficient). | Should |

### 6.5 Conversational Explainability Interface (Subsystem E)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-33 | Natural-language query interface (e.g., "Why did Buyer A reject Seller A's first offer?"). | Must |
| FR-34 | Explanations must be grounded in actual recorded episode data, not speculative LLM generation. | Must |
| FR-35 | FastAPI backend exposing REST + WebSocket endpoints for simulation control, dashboard data, and conversational queries. | Must |
| FR-36 | Next.js + Tailwind frontend with five sections: Marketplace, Agent Profiles, Negotiation Replay, Explainability, Experiment Lab. | Must |
| FR-37 | Negotiation replay: review full conversation and decision timeline for any episode. | Must |
| FR-38 | Strategy evolution timeline: compare an agent's behaviour at 100, 500, 1,000, 5,000 episodes. | Should |
| FR-39 | Plotly visualizations for reward curves, action distributions, market dynamics. | Should |

### 6.6 Data & Storage

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-40 | SQLite (development) / PostgreSQL (production) for episode logs, trajectories, and experiment metadata. | Must |
| FR-41 | Export capabilities: CSV/JSON for trajectories, PyTorch .pt for policy checkpoints. | Should |

---

## 7. Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| Performance | Negotiation episode simulation (no rendering) | ≥100 episodes/second |
| Performance | Conversational query response time | <5 seconds |
| Performance | Policy inference (single agent step) | <50 ms on GPU |
| Scalability | Support up to 10 agents (initial: 5) | — |
| Reliability | Deterministic seeding for reproducible experiments | — |
| Observability | Structured logging for all RL and XAI events | — |
| Security | Role-based access for dashboard (researcher vs. demo user) | Should |
| Deployability | Dockerized backend + frontend; one-command setup | Should |
| Maintainability | Type hints (Python) + TypeScript (frontend); ≥80% test coverage on RL engine | Should |

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEXUS Dashboard                         │
│  (Next.js + Tailwind + Plotly)                                  │
│  Marketplace │ Agent Profiles │ Replay │ Explainability │ Lab   │
└──────────────┴────────────────┴──────────┴────────────┴───────┘
                          │ HTTPS / WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  • Simulation control                                           │
│  • Episode / trajectory retrieval                               │
│  • Conversational query → explanation generator                 │
│  • Experiment run management                                    │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌───────────────┐ ┌─────────────────────────────────────────────┐
│   SQLite /    │ │           RL Engine (PyTorch)               │
│  PostgreSQL   │ │  • MAPPO (CTDE)                             │
│               │ │  • Independent PPO baseline                 │
│  Episode logs │ │  • Policy networks (actor/critic)           │
│  Trajectories │ │  • Reward functions                         │
│  Checkpoints  │ │  • GPU training / inference                 │
└───────────────┘ └──────────────┬───────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         PettingZoo Marketplace Environment (Gymnasium)          │
│  • 5 agents (2 buyers, 2 sellers, 1 market manager)             │
│  • State: sᵢ = [Bᵢ, Iᵢ, Pᵢ, Qᵢ, Hᵢ]                             │
│  • Actions: {Offer, Counteroffer, Accept, Reject, Wait, Withdraw}│
│  • Rule-based baselines                                         │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌───────────────┐ ┌─────────────────────────────────────────────┐
│  Explainability│ │      Conversational Negotiation Engine     │
│  Engine (XAI)  │ │  • BERT intent classification              │
│  • Trajectory  │ │  • Slot extraction (price, quantity)       │
│    recording   │ │  • Dialogue generation (template + LLM)    │
│  • Reward      │ │  • Offer validation                        │
│    attribution │ │  • RL policy → action (LLM does NOT decide)│
│  • Counterfactual│ │                                            │
│    rollouts    │ └─────────────────────────────────────────────┘
│  • Policy diff │
└───────────────┘
```

**Key architectural principle: CTDE.** During training, the MAPPO critic accesses the global state (all agents' observations). During execution, each agent's actor uses only its own permitted observation — simulating realistic information asymmetry.

**Key architectural principle: RL decides, NLP communicates.** The policy network selects {Offer, Counteroffer, Accept, Reject, Wait, Withdraw} and a price band. BERT parses incoming dialogue into intent + slots. A template/LLM generates the utterance. The LLM is never the primary decision-maker.

---

## 9. Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Multi-agent environment | PettingZoo + Gymnasium | Standard MARL API; interchangeable environments |
| RL algorithm | PyTorch + MAPPO | CTDE; mature, scalable on-policy MARL |
| Baseline | Independent PPO (PyTorch) | Ablation comparison |
| NLP | HuggingFace Transformers (BERT) | Intent classification + slot extraction |
| Dialogue generation | Template system + existing LLM | Grounded, controllable utterances |
| Explainability | Captum + SHAP (surrogate only) + custom counterfactual rollouts | Feature attribution must be complemented by policy probabilities and counterfactuals |
| Backend | FastAPI | Async REST + WebSocket; Python-native |
| Frontend | Next.js + Tailwind CSS | Reactive dashboard; easy deployment |
| Visualization | Plotly | Interactive reward curves, action distributions |
| Experiment tracking | TensorBoard / MLflow | Reproducible runs, metric comparison |
| Data storage | SQLite (dev) / PostgreSQL (prod) | Episode logs, trajectories, metadata |
| Containerization | Docker + Docker Compose | One-command setup |
| Compute | GPU (CUDA) for training; CPU acceptable for inference | — |

---

## 10. Data Model (Simplified)

### Episode
- episode_id (PK), start_time, end_time, config_hash, status, winner_agent_id

### Agent
- agent_id (PK), role, reward_function_config, initial_budget, initial_inventory, reservation_price

### TrajectoryStep
- step_id (PK), episode_id (FK), agent_id (FK), timestep, observation (JSON), action, action_probabilities (JSON), available_actions (JSON), reward (float), reward_components (JSON), next_observation (JSON), done (bool)

### PolicyCheckpoint
- checkpoint_id (PK), agent_id (FK), episode_count, model_path, performance_metrics (JSON)

### NegotiationDialogue
- dialogue_id (PK), episode_id (FK), turn_number, speaker_agent_id, utterance_text, structured_action (JSON), intent (from BERT), slots (JSON)

### Experiment
- experiment_id (PK), name, algorithm, config, start_time, end_time, results_summary (JSON)

---

## 11. API Design (FastAPI)

### Simulation Control
- `POST /api/v1/simulation/start` — Start a new negotiation episode
- `POST /api/v1/simulation/{episode_id}/step` — Advance one negotiation step
- `POST /api/v1/simulation/{episode_id}/reset` — Reset environment
- `GET /api/v1/simulation/{episode_id}/state` — Get current market state

### Trajectories & Explainability
- `GET /api/v1/episodes/{episode_id}/trajectory` — Full trajectory for an episode
- `GET /api/v1/episodes/{episode_id}/trajectory/{step_id}` — Single step detail
- `GET /api/v1/agents/{agent_id}/policy-diff?from=100&to=5000` — Compare policy across checkpoints
- `POST /api/v1/explain/counterfactual` — Run counterfactual rollout
- `GET /api/v1/explain/reward-attribution?episode_id=X&step_id=Y` — Reward decomposition

### Conversational Interface
- `POST /api/v1/query` — Natural-language question → evidence-backed explanation
  - Request: `{ question: string, episode_id?: string, agent_id?: string }`
  - Response: `{ answer: string, evidence: Evidence[], confidence: float }`

### Experiments
- `POST /api/v1/experiments` — Create and launch an experiment run
- `GET /api/v1/experiments/{id}/results` — Compare algorithm performance
- `GET /api/v1/experiments/compare` — Side-by-side MAPPO vs IPPO vs rule-based

---

## 12. Six-Week Implementation Roadmap

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| **Week 1** | Marketplace simulation | PettingZoo environment, 5 agents, state/action definitions, negotiation mechanics, rule-based baselines |
| **Week 2** | Reinforcement learning | Independent PPO baseline, MAPPO (CTDE), agent-specific reward functions, training loop, TensorBoard tracking |
| **Week 3** | Natural-language negotiation | BERT intent classifier + slot extractor, dialogue generation, action↔utterance mapping, validation layer |
| **Week 4** | Explainability engine | Trajectory recording, reward attribution, state-action analysis, policy evolution, counterfactual rollouts |
| **Week 5** | Conversational interface + dashboard | FastAPI backend, Next.js frontend (5 sections), negotiation replay, conversational Q&A, Plotly visualizations |
| **Week 6** | Experiments & evaluation | Run 4 experiments, compare MAPPO vs IPPO vs rule-based, generalization testing, final demo + results report |

---

## 13. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MAPPO vs rule-based average reward | +20% improvement | Experiment 1 |
| Intent classification accuracy | ≥85% | Held-out negotiation corpus |
| Query response time | <5 s | Dashboard conversational interface |
| Episode throughput | ≥100 eps/sec | Simulation benchmark |
| Reproducibility | Identical results with same seed | 10 runs, same seed |
| Explainability coverage | ≥5 investigation questions answerable | Dashboard explainability tab |

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MAPPO training instability | Medium | High | Use PPO clipping, entropy bonus, learning rate scheduling; monitor KL divergence |
| Non-stationarity (agents learn simultaneously) | High | Medium | Use centralized critic; evaluate against fixed opponents periodically |
| LLM hallucinating explanations | Medium | High | Ground all explanations in recorded trajectories; LLM only formats evidence |
| BERT misclassifying negotiation intent | Medium | Medium | Fine-tune on synthetic negotiation corpus; human-in-the-loop validation early |
| Reward hacking / loopholes | Medium | High | Distinguish strategic exploitation from bugs; reward-function audits in Experiment 3 |
| GPU availability | Low | Medium | CPU fallback for inference; smaller model for training if needed |

---

## 15. Out of Scope (Explicitly NOT Included)

- **LLM as primary decision-maker.** Agents never delegate negotiation choices to an LLM.
- **Training large language models.** Only fine-tuning a small BERT; using existing LLMs for explanation formatting only.
- **Real-world marketplace integration.** No connections to actual procurement or trading systems.
- **More than 10 agents.** Initial scope capped at 5 agents for meaningful experiments.
- **Continuous pricing at launch.** Discrete price bands to start; continuous pricing is a post-MVP extension.
- **Mobile apps.** Web dashboard only.

---

## 16. Open Questions for Stakeholders

1. **Reservation price configuration:** Should reservation prices be fixed per agent or dynamically adjusted during training?
2. **Price granularity:** Start with 10 price bands (e.g., ₹50–₹150 in ₹10 steps) or continuous?
3. **Experiment duration:** Is 5,000 episodes sufficient for convergence, or should we plan for 10,000+?
4. **Deployment target:** Local GPU workstation vs. cloud GPU instance?
5. **Baseline sophistication:** Should rule-based agents include tit-for-tat or only fixed strategies?

---

## 17. Appendices

### A. Example Negotiation Trace

| Turn | Speaker | Action | Structured Action | Utterance |
|------|---------|--------|-------------------|-----------|
| 1 | Seller A | Offer | {price: 100, quantity: 10} | "I can supply 10 units at ₹100 each." |
| 2 | Buyer A | Counteroffer | {price: 70, quantity: 10} | "I can purchase 10 units at ₹70 each." |
| 3 | Seller A | Counteroffer | {price: 90, quantity: 10} | "I can supply 10 units, but my minimum is ₹90 each." |
| 4 | Buyer A | Accept | {price: 85, quantity: 10} | "I can agree to ₹85 per unit if you deliver all 10 units." |

### B. Example Explainability Query

**Q:** "Why did Buyer A reject Seller A's first offer?"

**A:** Buyer A rejected the initial offer because its estimated expected return from counteroffering exceeded the expected return from accepting.

**Evidence:**
- Initial offer: ₹100/unit.
- Buyer's reservation price: ₹85/unit.
- Action probabilities at that state: Counteroffer 0.62, Accept 0.18, Reject 0.20.
- Reward components: immediate savings (−₹15/unit if accepted) vs. expected future savings from continued negotiation.

---

**End of PRD**

*Next step: Define the mathematical environment (state space, action space, observation space, reward functions, MAPPO training architecture) before frontend development begins.*