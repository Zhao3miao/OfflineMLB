"""Greedy random shooting on the current or next population snapshot."""
import copy
from collections import deque

import numpy as np

from env import CIOLoadBalancingEnv, RealDataCIOLoadBalancingEnv


class ClairvoyantGreedyPolicy:
    """Choose the best of N uniform actions using current or actual next snapshot.

    mode='current' freezes observed measurements; default mode='next' retains
    the original privileged lookahead. Ties use the first sampled candidate.

    Every candidate starts with the same CIO, associations, measurements and
    clock. Only read-only scenario/config data are shared between branches;
    Control state, including measurement caches and RNGs, is deep-copied.
    Historical plot coordinates are read-only during step; their deque
    containers are copied while the old coordinate values can be shared.
    The caller must execute the returned action once on the real environment.
    This finite, one-step search is not a long-horizon performance upper bound.
    """

    def __init__(self, seed=42, num_candidates=64, mode="next"):
        if not isinstance(num_candidates, int) or num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")
        if mode not in ("current", "next"):
            raise ValueError("mode must be current or next")
        self.num_candidates = num_candidates
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def current_reward(branch, action):
        """Use reported measurements and known control rules without time advance.

        No population refresh or physical RSRP remeasurement is allowed here.
        Actual execution still uses the normal env.step().
        """
        old_cio = branch.cio_matrix.copy()
        branch._apply_cio_action(action)
        branch.last_handover_baseline, branch.last_pingpong_baseline = (
            branch._get_hypothetical_connections(old_cio))
        branch._update_connections()
        branch._calculate_load()
        return branch._calculate_reward(branch._calculate_throughput(),
                                        branch._calculate_fairness(),
                                        branch._calculate_blocking())

    def select_action(self, environment):
        # Sharing scenario/config is valid for these deterministic replay envs.
        # Other env implementations must be audited before enabling sharing.
        if type(environment) not in (CIOLoadBalancingEnv, RealDataCIOLoadBalancingEnv):
            raise TypeError("Unsupported environment for clairvoyant replay")
        if environment.current_time >= environment.total_time - 1:
            raise RuntimeError("No next snapshot is available")
        if environment.fig is not None or environment.rendering:
            raise ValueError("Use an environment without active rendering")

        actions = self.rng.uniform(
            environment.action_space.low, environment.action_space.high,
            size=(self.num_candidates,) + environment.action_space.shape,
        ).astype(environment.action_space.dtype)
        rewards = np.empty(self.num_candidates, dtype=np.float64)
        for index, action in enumerate(actions):
            branch = copy.deepcopy(environment, memo={
                id(environment.scenario): environment.scenario,
                id(environment.config): environment.config,
                # step() only appends new lists here. Copying all old coordinate
                # lists for every candidate otherwise dominates long episodes.
                id(environment.device_trajectories): {
                    uid: deque(history, maxlen=history.maxlen)
                    for uid, history in environment.device_trajectories.items()
                },
            })
            if self.mode == "current":
                reward = self.current_reward(branch, action)
            else:
                _, reward, _, _, _ = branch.step(action)
            rewards[index] = reward
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Candidate rewards must be finite")
        selected = int(np.argmax(rewards))  # First sampled action wins ties.
        return actions[selected].copy(), {
            "selected_index": selected,
            "predicted_reward": float(rewards[selected]),
            "candidate_actions": actions,
            "candidate_rewards": rewards,
            "scoring_snapshot": self.mode,
        }
