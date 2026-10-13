"""Evaluate relative peak load using the already selected checkpoint."""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agent.attention_data import observation_to_attention_inputs
from utils.experiment_utils import (
    collect_targets, load_actor, load_environment, make_actor_step, read_json, set_cpu_threads, write_json,
)


def evaluate(args):
    set_cpu_threads(2)
    directory = Path(args.run_dir)
    summary = read_json(directory / "checkpoint_summary.json")
    checkpoint = Path(summary["selected_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    actor, norm, config, kind = load_actor(checkpoint, device)
    select = make_actor_step(actor, norm, config, kind, device)
    episodes = []
    for index, target in enumerate(collect_targets(args)):
        env = load_environment(target)
        try:
            observation, _ = env.reset(seed=args.seed + index)
            history = deque([observation_to_attention_inputs(env, observation)], maxlen=64)
            stds, excess, mean_loads, peaks = [], [], [], []
            done = False
            while not done:
                observation, _, terminal, truncated, info = env.step(select(history))
                history.append(observation_to_attention_inputs(env, observation))
                loads = np.asarray(env._get_bs_loads(), dtype=float)
                mean = float(loads.mean())
                stds.append(float(-info["load_balance"]))
                mean_loads.append(mean)
                peaks.append(float(loads.max()))
                if mean > 0:
                    excess.append(float(loads.max() / mean - 1))
                done = terminal or truncated
            episodes.append(dict(scenario_id=target["scenario_id"], mean_load_std=float(np.mean(stds)),
                mean_excess_peak_load=float(np.mean(excess)) if excess else 0.,
                mean_cell_load=float(np.mean(mean_loads)), mean_max_cell_load=float(np.mean(peaks))))
        finally:
            env.close()
    score = float(np.mean([row["mean_load_std"] for row in episodes]))
    if not np.isclose(score, summary["selected"]["test_mean_load_std"], rtol=1e-6, atol=1e-7):
        raise ValueError("Selected-checkpoint replay differs from saved test evaluation")
    write_json(args.output, dict(seed=args.seed, checkpoint_step=summary["selected"]["step"], episodes=episodes,
        mean_load_std=score, mean_excess_peak_load=float(np.mean([row["mean_excess_peak_load"] for row in episodes]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--env-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    evaluate(parser.parse_args())
