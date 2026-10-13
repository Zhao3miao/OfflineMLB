"""Evaluate static/random baselines and current-snapshot or clairvoyant greedy.

Writes one unified-format JSON per policy plus a summary, so the references use
the same episodes/mean_load_std structure as the learned-method evaluations.
Environment targets come from --env and/or --env-list (map YAMLs, processed MRO
windows, directories or JSON path lists).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from agent.clairvoyant_greedy import ClairvoyantGreedyPolicy
from utils.experiment_utils import collect_targets, evaluate_baseline, load_environment, run_episode, write_json

POLICIES = ("static_cio", "random", "clairvoyant_greedy")


def evaluate_greedy(targets, seed, max_steps, num_candidates, mode="next"):
    """Every candidate starts from the same state; mode sets snapshot access.

    This is a finite one-step reference, not a long-horizon upper bound.
    """
    episodes = []
    for index, target in enumerate(targets):
        environment = load_environment(target)
        try:
            step_seed = seed + index
            policy = ClairvoyantGreedyPolicy(seed=step_seed, num_candidates=num_candidates, mode=mode)

            def select(history, environment=environment, policy=policy):
                action, _ = policy.select_action(environment)
                return action

            row = run_episode(environment, select, step_seed, max_steps)
            row.update({"scenario_id": target["scenario_id"], "profile": target["profile"],
                        "kind": target["kind"]})
            episodes.append(row)
        finally:
            environment.close()
    values = [episode["mean_load_std"] for episode in episodes if np.isfinite(episode["mean_load_std"])]
    policy_name = "current_state_greedy" if mode == "current" else "clairvoyant_greedy"
    return {"policy": policy_name, "seed": seed, "episodes": episodes,
            "mean_load_std": float(np.mean(values)) if values else float("nan"),
            "std_across_scenarios": float(np.std(values)) if len(values) > 1 else 0.0,
            "evaluation_policy": policy_name, "num_episodes": len(episodes),
            "num_candidates": int(num_candidates), "scoring_snapshot": mode,
            "uses_future_snapshot": mode == "next"}


def evaluate(args):
    targets = collect_targets(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    policies = getattr(args, "policies", POLICIES)
    for policy in ("static_cio", "random"):
        if policy not in policies:
            continue
        result = evaluate_baseline(targets, policy, args.seed, args.max_steps)
        write_json(output_dir / (policy + ".json"), result)
        summary[policy] = result["mean_load_std"]
        print("%s mean_load_std=%.6f" % (policy, result["mean_load_std"]), flush=True)
    if "clairvoyant_greedy" in policies:
        greedy = evaluate_greedy(targets, args.seed, args.max_steps, args.num_candidates,
                                 mode=getattr(args, "greedy_mode", "next"))
        name = greedy["policy"]
        write_json(output_dir / (name + ".json"), greedy)
        summary[name] = greedy["mean_load_std"]
        print("%s mean_load_std=%.6f" % (name, greedy["mean_load_std"]), flush=True)
    # Both greedy modes may be evaluated in separate invocations.
    for name in ("static_cio", "random", "current_state_greedy", "clairvoyant_greedy"):
        path = output_dir / (name + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["seed"] != args.seed:
                raise ValueError("Baseline output directory contains another seed")
            summary[name] = saved["mean_load_std"]
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", help="Map YAML, directory, processed MRO directory or JSON list")
    parser.add_argument("--env-list", help="JSON file with a list of environment paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Zero evaluates full episodes")
    parser.add_argument("--num-candidates", type=int, default=64,
                        help="Candidate actions per step for the clairvoyant greedy reference")
    parser.add_argument("--greedy-mode", choices=("current", "next"), default="next",
                        help="current scores frozen observed measurements; next uses privileged lookahead")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES),
                        help="Select baseline families; greedy-mode controls the greedy information set")
    args = parser.parse_args(argv)
    if args.max_steps < 0 or args.num_candidates < 1:
        parser.error("max-steps must be nonnegative and num-candidates positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
