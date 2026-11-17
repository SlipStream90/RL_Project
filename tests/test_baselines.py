import numpy as np

from nexus.env import NexusMarketplaceEnv
from nexus.env.baselines import baseline_policy


def _run_episode(seed=42):
    env = NexusMarketplaceEnv()
    env.reset(seed=seed)
    pol = {a: baseline_policy(a, seed=1) for a in env.possible_agents}
    acts = []
    for _ in range(400):
        if not env.agents:
            break
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None)
            continue
        act = pol[a].act(env.observe(a), env.action_masks(a), np.random.default_rng(7))
        assert env.action_masks(a)["type"][int(act[0])]
        acts.append((a, tuple(int(x) for x in act)))
        env.step(act)
    return acts, env.get_trajectory()


def test_baselines_deterministic():
    a1, t1 = _run_episode()
    a2, t2 = _run_episode()
    assert a1 == a2
    assert len(t1) == len(t2) > 0


def test_baseline_policy_all_agents():
    for aid in ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"]:
        p = baseline_policy(aid, seed=0)
        assert p is not None
