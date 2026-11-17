# NEXUS — Mathematical Environment Specification

**Document:** Technical Environment Spec  
**Version:** 1.0  
**Date:** November 2026  
**Status:** Draft for Implementation  

This document defines the mathematical foundation of the NEXUS marketplace environment: state space, action space, observation space, reward functions, transition dynamics, and MAPPO training architecture. It is intended to be implementable directly in PettingZoo + PyTorch.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of agents (initial: $N=5$) |
| $\mathcal{A} = \{1,\dots,N\}$ | Agent index set |
| $t$ | Discrete timestep within an episode |
| $T$ | Maximum episode length (timesteps) |
| $s_i^t$ | Local observation of agent $i$ at timestep $t$ |
| $S^t$ | Global state at timestep $t$ (critic-only) |
| $a_i^t$ | Action of agent $i$ at timestep $t$ |
| $r_i^t$ | Reward received by agent $i$ at timestep $t$ |
| $\pi_{\theta_i}$ | Actor policy of agent $i$ (parameters $\theta_i$) |
| $V_{\phi}$ | Centralized critic (parameters $\phi$) |
| $\gamma$ | Discount factor |

---

## 2. Agent Roles and Private Configuration

Five agents, each with a distinct role and private configuration vector $c_i$:

| Agent | ID | Role | Primary Objective |
|-------|----|------|-------------------|
| Buyer A | 1 | Purchaser | Buy resources cheaply (minimize cost) |
| Buyer B | 2 | Purchaser | Secure resources quickly (minimize time) |
| Seller A | 3 | Supplier | Maximize profit (margin per unit) |
| Seller B | 4 | Supplier | Maximize sales volume (units moved) |
| Market Manager | 5 | Inventory controller | Maintain market stability (price variance, transaction rate) |

### Private configuration $c_i$

Each agent is initialized with a private configuration drawn from configured distributions:

$$
c_i = \{B_i^{(0)},\, I_i^{(0)},\, R_i,\, Q_i^{\text{req}},\, \rho_i,\, \mathcal{P}_i^{\text{priv}}\}
$$

| Field | Description | Domain | Example |
|-------|-------------|--------|---------|
| $B_i^{(0)}$ | Starting capital (budget) | $\mathbb{R}^+$ | Buyer A: ₹10,000 |
| $I_i^{(0)}$ | Starting inventory (units) | $\mathbb{N}_0$ | Seller A: 100 units |
| $R_i$ | Reservation price (min acceptable for seller / max acceptable for buyer) | $\mathbb{R}^+$ | Buyer A: ₹85/unit |
| $Q_i^{\text{req}}$ | Required quantity per episode | $\mathbb{N}^+$ | Buyer A: 10 units |
| $\rho_i$ | Risk tolerance ($0$ = risk-averse, $1$ = risk-seeking) | $[0,1]$ | Seller A: 0.3 |
| $\mathcal{P}_i^{\text{priv}}$ | Private information (known only to agent $i$) | varies | Seller A knows true stock; buyers see partial market info |

**Important:** Agents do not observe other agents' $c_j$ directly. They must infer intentions through negotiation behavior.

---

## 3. State Space

### 3.1 Local observation $s_i^t$ (actor input)

Each agent observes only its own permitted state:

$$
s_i^t = \big[B_i^t,\, I_i^t,\, P_i^t,\, Q_i^t,\, H_i^t,\, M_i^t\big] \in \mathbb{R}^{d_s}
$$

| Component | Symbol | Description | Domain |
|-----------|--------|-------------|--------|
| Available budget | $B_i^t$ | Remaining capital | $\mathbb{R}^+$ |
| Current inventory | $I_i^t$ | Units held | $\mathbb{N}_0$ |
| Current offer price | $P_i^t$ | Price on the table (last offer/counteroffer) | $\mathbb{R}^+ \cup \{0\}$ (0 = no active offer) |
| Required quantity remaining | $Q_i^t$ | Units still needed/sold this episode | $\mathbb{N}_0$ |
| Negotiation history | $H_i^t$ | Fixed-length feature vector (see below) | $\mathbb{R}^{d_h}$ |
| Market context | $M_i^t$ | Public market signals visible to agent $i$ | $\mathbb{R}^{d_m}$ |

#### Negotiation history features $H_i^t$

Fixed-length window of the last $k$ turns:

$$
H_i^t = \big[\text{offer\_prices}_{t-k:t},\, \text{offer\_quantities}_{t-k:t},\, \text{intent\_onehot}_{t-k:t},\, \text{time\_since\_last\_offer}\big]
$$

- `offer_prices`: last $k$ offer prices seen by agent $i$ (padded with 0)
- `offer_quantities`: last $k$ quantities (padded with 0)
- `intent_onehot`: one-hot encoding of last $k$ intents from counterparty {OFFER, COUNTEROFFER, ACCEPT, REJECT, WAIT, WITHDRAW}
- `time_since_last_offer`: timesteps since the last offer/counteroffer

Suggested: $k=5$, $d_h = 5k + 5k + 6k + 1 = 61$

#### Market context $M_i^t$

Public signals visible to agent $i$ (depends on information asymmetry):

- Average transaction price over last $m$ completed deals
- Number of active buyers/sellers
- Time remaining in episode ($T - t$)
- Inventory scarcity signal (total market inventory / total demand)

### 3.2 Global state $S^t$ (critic input)

The centralized MAPPO critic observes the full joint state:

$$
S^t = \big[\, s_1^t, s_2^t, \dots, s_N^t,\, \text{market\_clearing\_price}^t,\, \text{total\_inventory}^t,\, \text{total\_demand}^t,\, \text{episode\_progress}^t \,\big]
$$

This includes all agents' private configurations and the full negotiation transcript — information no single agent possesses. Used **only** during training for the value function.

---

## 4. Action Space

### 4.1 Discrete action space (recommended for MVP)

Each agent selects a discrete action:

$$
a_i^t = (\text{type}_i^t,\, \text{price\_band}_i^t,\, \text{quantity}_i^t) \in \mathcal{A}_i
$$

#### Action type

$$
\text{type}_i^t \in \{\text{OFFER, COUNTEROFFER, ACCEPT, REJECT, WAIT, WITHDRAW}\}
$$

| Action | Semantics | Valid when |
|--------|-----------|------------|
| OFFER | Propose a new deal | No active offer from counterparty, or after REJECT |
| COUNTEROFFER | Respond to active offer with new price/qty | Active offer on table |
| ACCEPT | Accept current active offer | Active offer on table, within budget/inventory |
| REJECT | Reject current active offer | Active offer on table |
| WAIT | Pass turn, no state change | Always |
| WITHDRAW | Exit negotiation permanently | Always (episode ends for this agent) |

#### Price bands

Discrete pricing to start (continuous pricing is a post-MVP extension):

$$
\text{price\_band} \in \{1, 2, \dots, n_p\}
$$

Mapped to actual price via:

$$
\text{price} = p_{\min} + (\text{price\_band} - 1) \cdot \Delta p
$$

**Recommended:** ₹50–₹200 in ₹10 steps → $n_p = 16$ bands, $\Delta p = ₹10$.

#### Quantity

$$
\text{quantity} \in \{1, 2, \dots, Q_i^{\text{req}}\}
$$

Capped by inventory (sellers) or budget/affordability (buyers).

### 4.2 Action space size

$$
|\mathcal{A}_i| = 6 \times n_p \times Q_i^{\text{req}}
$$

With $n_p = 16$ and $Q_i^{\text{req}} = 10$: $|\mathcal{A}_i| = 960$ discrete actions per agent.

**Implementation:** Use `gymnasium.spaces.MultiDiscrete([6, n_p, Q_max])` or flatten to `Discrete(6 * n_p * Q_max)`.

---

## 5. Observation Space

### 5.1 Actor observation (per agent)

$$
\mathcal{O}_i = \text{Box}(\text{low},\, \text{high},\, \text{shape}=(d_s,),\, \text{dtype}=\text{float32})
$$

All components normalized to $[0,1]$ or standardized before network input.

### 5.2 Critic observation (global)

$$
\mathcal{O}_{\text{critic}} = \text{Box}(\text{low},\, \text{high},\, \text{shape}=(N \cdot d_s + d_{\text{market}},),\, \text{dtype}=\text{float32})
$$

---

## 6. Reward Functions

Rewards are **agent-specific** and aligned with each role's objective. This is critical — identical rewards remove strategic diversity.

Reward is computed at episode end (terminal reward) and/or per-step (shaping). We use a hybrid:

$$
r_i^t = r_i^{\text{step}} + r_i^{\text{terminal}}
$$

### 6.1 Buyer reward

Objective: acquire required quantity at low cost, quickly.

$$
r_{\text{buyer}} = w_1 \cdot \underbrace{\big(1 - \tfrac{p_{\text{avg}}}{R_{\text{buyer}}}\big)}_{\text{price efficiency}} + w_2 \cdot \underbrace{\big(1 - \tfrac{t_{\text{deal}}}{T}\big)}_{\text{speed bonus}} + w_3 \cdot \underbrace{\mathbb{I}[\text{deal completed}]}_{\text{completion}} - w_4 \cdot \underbrace{\mathbb{I}[\text{budget exceeded}]}_{\text{constraint violation}}
$$

- $p_{\text{avg}}$: average price paid per unit
- $R_{\text{buyer}}$: reservation price (max acceptable)
- $t_{\text{deal}}$: timestep when deal completed
- $w_1 + w_2 + w_3 + w_4 = 1$

**Buyer A (price-focused):** $w = [0.5, 0.2, 0.2, 0.1]$  
**Buyer B (speed-focused):** $w = [0.2, 0.5, 0.2, 0.1]$

### 6.2 Seller reward

Objective varies by seller type.

**Seller A (profit-focused):**

$$
r_{\text{seller A}} = w_1 \cdot \underbrace{\tfrac{p_{\text{avg}} - c_{\text{unit}}}{R_{\text{seller}} - c_{\text{unit}}}}_{\text{margin}} + w_2 \cdot \mathbb{I}[\text{deal completed}] - w_3 \cdot \underbrace{\tfrac{I^{\text{unsold}}}{I^{(0)}}}_{\text{inventory penalty}}
$$

**Seller B (volume-focused):**

$$
r_{\text{seller B}} = w_1 \cdot \underbrace{\tfrac{Q_{\text{sold}}}{Q^{\text{req}}}}_{\text{fill rate}} + w_2 \cdot \mathbb{I}[\text{deal completed}] - w_3 \cdot \underbrace{\big|p_{\text{avg}} - R_{\text{seller}}\big|}_{\text{price deviation penalty}}
$$

### 6.3 Market Manager reward

Objective: market stability.

$$
r_{\text{manager}} = w_1 \cdot \underbrace{\big(1 - \tfrac{\sigma_p}{\mu_p}\big)}_{\text{price stability (low CV)}} + w_2 \cdot \underbrace{\tfrac{N_{\text{completed}}}{N_{\text{attempted}}}}_{\text{transaction rate}} - w_3 \cdot \underbrace{\mathbb{I}[\text{market collapse}]}_{\text{no deals}}
$$

Where $\sigma_p / \mu_p$ is the coefficient of variation of transaction prices.

### 6.4 Reward shaping (per-step)

Small per-step rewards to guide learning:

- +0.01 for making a valid offer/counteroffer (encourages activity)
- -0.01 for WAIT (discourages stalling, small)
- -0.1 for WITHDRAW (penalty for abandoning)
- 0 for ACCEPT/REJECT (outcome captured in terminal reward)

**Constraint violations** (accepting above reservation, selling below cost, exceeding budget) receive large negative terminal reward and terminate the agent's participation.

---

## 7. Transition Dynamics & Negotiation Protocol

### 7.1 Turn-taking

The environment follows a PettingZoo AEC (Agent Environment Cycle) order:

1. Agent $i$ is selected (round-robin or based on who has the active offer)
2. Agent $i$ observes $s_i^t$ and selects $a_i^t$
3. Environment applies action, updates state, computes $r_i^t$
4. Next agent selected

### 7.2 Negotiation state machine

```
State: IDLE
  → Agent OFFER → State: OFFER_ACTIVE (price P, quantity Q, offerer)
  
State: OFFER_ACTIVE
  → Counterparty ACCEPT → Deal completed → State: DEAL_DONE (episode may continue for other agents)
  → Counterparty COUNTEROFFER (P', Q') → State: OFFER_ACTIVE (new price, new offerer)
  → Counterparty REJECT → State: IDLE (offer cleared)
  → Counterparty WAIT → State: OFFER_ACTIVE (unchanged, turn passes)
  → Counterparty WITHDRAW → State: WITHDRAWN (agent out)
  
State: DEAL_DONE / WITHDRAWN
  → Episode continues until all agents done or T reached
```

### 7.3 Offer validity constraints

An OFFER or COUNTEROFFER $(p, q)$ by agent $i$ is valid iff:

**Buyer:**
- $p \cdot q \leq B_i^t$ (can afford)
- $q \leq Q_i^t$ (doesn't need more than required)
- $p \leq R_i^{\text{buyer}}$ (at or below reservation — though learning may violate this with penalty)

**Seller:**
- $q \leq I_i^t$ (has inventory)
- $p \geq c_{\text{unit}}$ (above unit cost, or penalty applied)

Invalid actions are masked by the environment (action masking) and receive a small negative reward.

### 7.4 Episode termination

An episode ends when:
- All agents have completed deals or withdrawn, OR
- Timestep $t = T$ (max episode length reached), OR
- Market collapse (no deals after $k$ consecutive WAITs from all agents)

### 7.5 Pseudocode: environment step

```python
def step(self, action):
    agent = self.agent_selection  # current agent in AEC order
    action_type, price_band, quantity = self.discretize(action)
    
    # Action masking
    if not self.is_valid(agent, action_type, price_band, quantity):
        self.rewards[agent] = -0.1  # invalid action penalty
        self._advance_turn()
        return
    
    # Execute action
    if action_type == OFFER:
        self.active_offer = (agent, price, quantity)
        self.rewards[agent] = 0.01  # small activity bonus
    
    elif action_type == COUNTEROFFER:
        self.active_offer = (agent, price, quantity)
        self.rewards[agent] = 0.01
    
    elif action_type == ACCEPT:
        deal = self.active_offer
        self._finalize_deal(deal)
        self.rewards[agent] = 0.0  # terminal reward handles outcome
        self.active_offer = None
    
    elif action_type == REJECT:
        self.active_offer = None
        self.rewards[agent] = 0.0
    
    elif action_type == WAIT:
        self.rewards[agent] = -0.01  # small stalling penalty
    
    elif action_type == WITHDRAW:
        self.withdrawn.add(agent)
        self.rewards[agent] = -0.1
    
    # Record trajectory for explainability
    self._record_trajectory(agent, action, reward, next_obs)
    
    # Advance turn
    self._advance_turn()
    
    # Check termination
    if self.all_done() or self.t >= self.T:
        self._compute_terminal_rewards()
        self.dones = {a: True for a in self.agents}
```

---

## 8. Information Asymmetry (Experiment 3 Support)

To study exploitation, agents receive **partial observations**:

| Information | Buyer A | Buyer B | Seller A | Seller B | Market Manager |
|-------------|---------|---------|----------|----------|----------------|
| Own budget | ✓ | ✓ | ✗ | ✗ | ✓ (aggregate) |
| Own inventory | ✗ | ✗ | ✓ | ✓ | ✓ (total) |
| Own reservation | ✓ | ✓ | ✓ | ✓ | ✗ |
| Counterparty reservation | ✗ | ✗ | ✗ | ✗ | ✗ |
| True market inventory | Partial | Partial | Full | Partial | Full |
| Competitor identities | ✗ | ✗ | ✗ | ✗ | ✓ |
| Past transaction prices | Last $k$ only | Last $k$ only | Last $k$ only | Last $k$ only | Full history |

Seller A knows true inventory; buyers see a noised version: $\hat{I}^{\text{market}} = I^{\text{market}} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

---

## 9. MAPPO Training Architecture (CTDE)

### 9.1 Centralized Training, Decentralized Execution

```
Training (centralized critic):
  Input to critic: S^t (global state) + all actions a_1^t ... a_N^t
  Output: V_phi(S^t) — value of global state
  
Execution (decentralized actors):
  Input to actor i: s_i^t (local observation only)
  Output: pi_theta_i(a | s_i^t) — action distribution for agent i
```

### 9.2 Network architectures

**Actor (per agent, shared or separate):**

```
Input: s_i (d_s) → MLP(256, 256) → 
  Branch A: Discrete action type logits (6)
  Branch B: Price band logits (n_p)  
  Branch C: Quantity logits (Q_max)
Output: Categorical distributions over each branch
```

**Critic (single, centralized):**

```
Input: S (global) → MLP(512, 256, 128) → scalar V(S)
```

Can use parameter sharing across agents of same role (buyers share actor weights, sellers share actor weights) or fully separate. Recommended: separate for MVP to preserve role diversity.

### 9.3 MAPPO loss

For each agent $i$, with advantage estimate $\hat{A}_i^t$:

$$
L^{\text{CLIP}}(\theta_i) = \mathbb{E}_t \left[ \min\left( \frac{\pi_{\theta_i}(a_i^t|s_i^t)}{\pi_{\theta_i^{\text{old}}}(a_i^t|s_i^t)} \hat{A}_i^t,\; \text{clip}\left(\frac{\pi_{\theta_i}}{\pi_{\theta_i^{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i^t \right) \right]
$$

$$
L^{\text{VF}}(\phi) = \mathbb{E}_t \left[ \left( V_{\phi}(S^t) - \hat{V}_t \right)^2 \right]
$$

$$
L^{\text{ENT}}(\theta_i) = \mathbb{E}_t \left[ \mathcal{H}\left(\pi_{\theta_i}(\cdot|s_i^t)\right) \right]
$$

Total loss:

$$
L = \sum_i \left( L^{\text{CLIP}}_i + c_1 L^{\text{VF}} - c_2 L^{\text{ENT}}_i \right)
$$

### 9.4 Advantage estimation (GAE)

$$
\delta_i^t = r_i^t + \gamma V_{\phi}(S^{t+1}) - V_{\phi}(S^t)
$$

$$
\hat{A}_i^t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_i^{t+l}
$$

### 9.5 Training loop

```
for episode in range(num_episodes):
    env.reset()
    trajectories = []
    
    while not done:
        for agent in env.agent_iter():
            obs = env.observe(agent)          # s_i (local)
            global_obs = env.get_global_state()  # S (critic)
            
            action, log_prob = actor[agent].sample(obs)
            value = critic(global_obs)
            
            env.step(action)
            reward = env.rewards[agent]
            
            trajectories.append(Transition(
                agent=agent, obs=obs, global_obs=global_obs,
                action=action, log_prob=log_prob, value=value,
                reward=reward, done=done
            ))
    
    # Compute advantages, update PPO epochs
    update_ppo(trajectories)
    
    # Save checkpoint every N episodes
    if episode % checkpoint_interval == 0:
        save_checkpoint(episode)
```

---

## 10. Explainability Data Requirements

To support the explainability engine, each transition records:

```python
@dataclass
class Transition:
    episode_id: str
    timestep: int
    agent_id: int
    observation: dict          # s_i^t (normalized + raw)
    global_state: dict         # S^t (critic input)
    action: tuple              # (type, price_band, quantity)
    action_log_prob: float     # log pi(a|s)
    action_probs: dict         # full distribution over all actions
    available_actions: list    # masked valid actions
    reward: float              # r_i^t
    reward_components: dict    # decomposed reward (price, speed, completion, etc.)
    next_observation: dict     # s_i^(t+1)
    done: bool
    utterance: str             # generated natural language
    intent: str                # BERT-classified intent of incoming message
    slots: dict                # extracted price, quantity
    checkpoint_id: int         # policy version at this timestep
```

This enables:
- **Reward attribution:** decompose $r_i^t$ by component
- **State-action analysis:** correlate $s_i^t$ features with $a_i^t$
- **Policy analysis:** compare $\pi_{\theta_i}$ across checkpoints
- **Counterfactuals:** replay with alternate actions using recorded $S^t$

---

## 11. Open Implementation Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Observation normalization | Min-max to [0,1] vs. standardization | Min-max to [0,1] (simpler, bounded) |
| Parameter sharing | Separate actors per role vs. shared | Separate per role (MVP) |
| Price bands | 10 bands (₹10 step) vs. 16 bands (₹10 step, ₹50–₹200) | 16 bands |
| Episode length | Fixed T=100 vs. variable | Fixed T=100 timesteps |
| History window | k=3 vs. k=5 | k=5 |
| Critic input | Full global state vs. state + actions | Full global state (simpler) |

---

**Next step after this spec:** Implement the PettingZoo environment class (`NexusMarketplaceEnv`) with the state/action/reward definitions above, then wire up MAPPO training.
