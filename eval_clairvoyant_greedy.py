"""Evaluate future-aware greedy search on lifecycle/map YAML or MRO snapshots.

Examples:
  python eval_clairvoyant_greedy.py --scenario-dir outputs/map_lopo_scenarios_v1
  python eval_clairvoyant_greedy.py --processed-dir processed_mro/18_45

Candidates are legal CIO increments (not absolute CIO settings). Only the
selected action advances the live environment. No checkpoint is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agent.clairvoyant_greedy import ClairvoyantGreedyPolicy
from env import CIOLoadBalancingEnv, RealDataCIOLoadBalancingEnv


def evaluate_policy(environment, seed=42, num_candidates=64, max_steps=0):
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative (0 means full episode)")
    environment.reset(seed=seed)
    policy = ClairvoyantGreedyPolicy(seed=seed, num_candidates=num_candidates)
    rows, candidate_actions, candidate_rewards = [], [], []
    started = time.monotonic()
    while environment.current_time < environment.total_time - 1:
        if max_steps and len(rows) >= max_steps:
            break
        action, decision = policy.select_action(environment)
        _, reward, terminated, truncated, info = environment.step(action)
        error = abs(float(reward) - decision["predicted_reward"])
        if error > 1e-10 or not np.isfinite(reward):
            raise RuntimeError("Executed reward differs from the selected branch")
        scores = decision["candidate_rewards"]
        candidate_actions.append(decision["candidate_actions"])
        candidate_rewards.append(scores)
        rows.append({
            "time_step": int(environment.current_time),
            "timestamp": info.get("timestamp"),
            "action": action.tolist(),
            "selected_index": decision["selected_index"],
            "predicted_reward": decision["predicted_reward"],
            "reward": float(reward),
            "load_std": float(-info["load_balance"]),
            "active_ue_count": int(info["active_ue_count"]),
            "handover_count": int(info["handover_count"]),
            "candidate_reward_mean": float(np.mean(scores)),
            "candidate_reward_range": float(np.ptp(scores)),
            "unique_candidate_rewards": int(len(np.unique(scores))),
        })
        if terminated or truncated:
            break
    if not rows:
        raise ValueError("Evaluation requires at least two population snapshots")
    result = {
        "policy": "clairvoyant_greedy",
        "seed": int(seed), "num_candidates": int(num_candidates),
        "reward_timing": "next_population", "lookahead_steps": 1,
        "candidate_sampling": "uniform_legal_action_box",
        "tie_break": "first_sampled", "max_steps": max_steps,
        "steps": len(rows),
        "full_episode": environment.current_time == environment.total_time - 1,
        "mean_load_std": float(np.mean([r["load_std"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "total_handovers": sum(r["handover_count"] for r in rows),
        "mean_candidate_reward_range": float(np.mean([
            r["candidate_reward_range"] for r in rows])),
        "fraction_steps_equal_candidate_rewards": float(np.mean([
            r["unique_candidate_rewards"] == 1 for r in rows])),
        "elapsed_seconds": time.monotonic() - started,
        "trace": rows,
    }
    return result, {
        "candidate_actions": np.stack(candidate_actions),
        "candidate_rewards": np.stack(candidate_rewards),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path, help="One lifecycle/map YAML")
    source.add_argument("--scenario-dir", type=Path, help="All YAML files in a directory")
    source.add_argument("--processed-dir", type=Path, help="One preprocessed MRO window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/clairvoyant_greedy"))
    args = parser.parse_args()
    paths = ([args.processed_dir] if args.processed_dir else
             [args.scenario] if args.scenario else
             sorted(set(args.scenario_dir.glob("*.yaml")) |
                    set(args.scenario_dir.glob("*.yml"))))
    if not paths:
        parser.error("No scenario files found")
    # A fresh run directory prevents accidentally mixing partial/old results.
    output = args.output_dir / ("seed_%d" % args.seed)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, path in enumerate(paths):
        env = (RealDataCIOLoadBalancingEnv(path) if args.processed_dir else
               CIOLoadBalancingEnv(str(path)))
        try:
            # Same action RNG seed per episode, independent of directory order.
            result, arrays = evaluate_policy(env, args.seed, args.num_candidates, args.max_steps)
            result["source"] = str(path.resolve())
            result["scenario_id"] = env.config.get("scenario_id", path.stem)
            result["population_profile"] = env.config.get("population_config", {}).get(
                "profile", "mro" if args.processed_dir else "unspecified")
            if path.is_file():
                result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            name = "%03d_%s" % (index, path.stem)
            np.savez_compressed(str(output / (name + "_candidates.npz")), **arrays)
            result["candidate_file"] = name + "_candidates.npz"
            (output / (name + ".json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {k: v for k, v in result.items() if k != "trace"}
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        finally:
            env.close()
    profiles = sorted(set(r["population_profile"] for r in summaries))
    aggregate = {
        "episodes": summaries,
        "mean_load_std": float(np.mean([r["mean_load_std"] for r in summaries])),
        "by_profile": {profile: {
            "num_episodes": sum(r["population_profile"] == profile for r in summaries),
            "mean_load_std": float(np.mean([r["mean_load_std"] for r in summaries
                                             if r["population_profile"] == profile])),
        } for profile in profiles},
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
