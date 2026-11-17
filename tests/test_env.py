import numpy as np
import pytest

from nexus.env import NexusMarketplaceEnv
from nexus.env.types import D_GLOBAL, D_S


def run_random_episode(seed=42, max_actions=500):
    env = NexusMarketplaceEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    while env.agents and steps < max_actions:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        masks = env.action_masks(agent)
        # sample a valid type then valid band/qty indices
        valid_types = np.flatnonzero(masks["type"])
        t = int(rng.choice(valid_types))
        b = int(rng.integers(0, 16))
        q = int(rng.integers(0, 10))
        env.step(np.array([t, b, q], dtype=np.int64))
        steps += 1
    traj = env.get_trajectory()
    return env, traj, steps


def test_agents_and_spaces():
    env = NexusMarketplaceEnv()
    assert env.possible_agents == ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"]
    env.reset(seed=0)
    for a in env.possible_agents:
        assert env.observation_space(a).shape == (D_S,)
        assert list(env.action_space(a).nvec) == [6, 16, 10]
    assert env.get_global_state().shape == (D_GLOBAL,)


def test_episode_terminates_and_trajectory():
    env, traj, steps = run_random_episode()
    assert steps <= 500
    assert len(traj) == steps
    assert all(t.observation.shape == (D_S,) for t in traj)
    assert all(set(t.reward_components.keys()) for t in traj)


def test_deterministic_seeding():
    _, t1, _ = run_random_episode(seed=7)
    _, t2, _ = run_random_episode(seed=7)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.action == b.action
        assert a.reward == b.reward
        assert np.allclose(a.observation, b.observation)


def test_invalid_action_penalty():
    env = NexusMarketplaceEnv()
    env.reset(seed=1)
    # buyer_a first: ACCEPT with no active offer is invalid
    agent = env.agent_selection
    env.step(np.array([2, 0, 0], dtype=np.int64))
    assert env.infos[agent].get("invalid") is True


def test_mask_blocks_accept_without_offer():
    env = NexusMarketplaceEnv()
    env.reset(seed=1)
    m = env.action_masks(env.agent_selection)
    assert not m["type"][2]  # ACCEPT
    assert m["type"][0]  # OFFER


def test_snapshot_roundtrip():
    env = NexusMarketplaceEnv()
    env.reset(seed=3)
    env.step(np.array([4, 0, 0], dtype=np.int64))
    snap = env.get_state()
    obs1 = env.observe(env.agent_selection).copy()
    env.step(np.array([4, 0, 0], dtype=np.int64))
    env.set_state(snap)
    obs2 = env.observe(env.agent_selection)
    assert np.allclose(obs1, obs2)


def test_offer_accept_flow():
    env = NexusMarketplaceEnv()
    env.reset(seed=5)
    # walk round-robin until a seller offers then a buyer accepts if possible
    rng = np.random.default_rng(0)
    for _ in range(200):
        if not env.agents:
            break
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        masks = env.action_masks(agent)
        if masks["type"][2]:
            env.step(np.array([2, 0, 0], dtype=np.int64))
        elif masks["type"][0]:
            env.step(np.array([0, 8, 4], dtype=np.int64))
        else:
            env.step(np.array([4, 0, 0], dtype=np.int64))
    # no crash; trajectory recorded
    assert len(env.get_trajectory()) > 0
