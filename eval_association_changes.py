"""Evaluate association changes using an already selected public checkpoint."""
import argparse
from collections import deque
import csv
import math
from pathlib import Path

import numpy as np
import torch

from agent.attention_data import observation_to_attention_inputs
from utils.experiment_utils import (
    collect_targets, load_actor, load_environment, make_actor_step, read_json,
    set_cpu_threads, sha256, write_json,
)

SCENES = {"map": "Map", "site_a": "Site A", "site_b": "Site B", "site_c": "Site C"}
METHODS = {"sac_h1": "SAC H=1", "sac_h5": "SAC H=5", "link_mvd_h5": "Link+MVD H=5"}
SEEDS = (42, 233, 666)
COUNTS = (
    "changed_vs_previous_cio", "changed_vs_zero_cio", "changed_vs_reference",
    "previous_vs_zero_cio", "zero_cio_vs_reference",
)
METRICS = ("mean_load_std",) + tuple("mean_" + name for name in COUNTS)


def summarize_episode(records):
    if not records:
        raise ValueError("No evaluation steps")
    eligible = [row for row in records if row["eligible_users"] > 0]
    if not eligible:
        raise ValueError("No observable users for association diagnostics")
    users = sum(row["eligible_users"] for row in records)
    result = dict(steps=len(records), eligible_user_observations=users,
                  steps_with_eligible_users=len(eligible))
    for name, field in (("mean_load_std", "load_std"), ("mean_cell_load", "mean_cell_load"),
                        ("mean_max_cell_load", "mean_max_cell_load")):
        result[name] = sum(row[field] for row in records) / len(records)
    for name in COUNTS:
        count = sum(row[name] for row in records)
        result["mean_" + name] = sum(row[name] / row["eligible_users"] for row in eligible) / len(eligible)
        result["pooled_" + name] = count / users
        result["total_" + name] = count
    return result


def evaluate_checkpoint(checkpoint, targets, seed, device, max_steps=0):
    actor, norm, config, kind = load_actor(Path(checkpoint), torch.device(device))
    select = make_actor_step(actor, norm, config, kind, torch.device(device))
    episodes = []
    for index, target in enumerate(targets):
        env = load_environment(target)
        env.record_association_diagnostics = True
        try:
            observation, _ = env.reset(seed=seed + index)
            history = deque([observation_to_attention_inputs(env, observation)], maxlen=64)
            records = []
            done = False
            while not done and (max_steps <= 0 or len(records) < max_steps):
                observation, reward, terminal, truncated, info = env.step(select(history))
                history.append(observation_to_attention_inputs(env, observation))
                counts = info["association_diagnostics"]
                n = counts["eligible_users"]
                if n < 0 or any(not 0 <= counts[name] <= n for name in COUNTS):
                    raise ValueError("Invalid association-change counts")
                loads = np.asarray(env._get_bs_loads(), dtype=np.float64)
                if not np.isfinite(loads).all() or (loads < 0).any():
                    raise ValueError("Invalid offered loads")
                records.append(dict(counts, time_step=int(info["time_step"]),
                                    load_std=-float(info["load_balance"]), reward=float(reward),
                                    mean_cell_load=float(loads.mean()), mean_max_cell_load=float(loads.max()),
                                    cio_change_mean_absolute=float(info["cio_change"])))
                done = terminal or truncated
            row = summarize_episode(records)
            row.update(scenario_id=target["scenario_id"], kind=target["kind"], step_records=records,
                       reference_kind=("snapshot_recorded_serving_cell" if target["kind"] == "mro"
                                       else "simulator_pre_update_connection"))
            episodes.append(row)
        finally:
            env.close()
    return dict(seed=seed, episodes=episodes, num_episodes=len(episodes),
                **{name: float(np.mean([row[name] for row in episodes])) for name in METRICS})


def selected_inputs(directory, seed):
    directory = Path(directory)
    summary = read_json(directory / "checkpoint_summary.json")
    if summary["seed"] != seed or summary.get("selection_metric") != "min_train_mean_load_std":
        raise ValueError("Run seed or checkpoint-selection rule differs")
    best = min(summary["checkpoints"], key=lambda row: (row["train_mean_load_std"], row["step"]))
    if summary["selected"] != best:
        raise ValueError("Checkpoint was not selected by training performance")
    checkpoint = Path(summary["selected_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    if summary.get("model_sha256") != sha256(checkpoint):
        raise ValueError("Selected checkpoint checksum differs")
    reference = read_json(directory / "evaluations" / ("step_%d_test.json" % best["step"]))
    return summary, checkpoint, reference


def finalize_result(result, summary, checkpoint, reference, env_list):
    """Keep portable metadata and verify every replay against the selected run."""
    episodes = result["episodes"]
    if (result["seed"] != summary["seed"] or len(episodes) != len(reference["episodes"]) or
            reference["seed"] != result["seed"]):
        raise ValueError("Evaluation seed or test-environment count differs")
    errors = []
    for actual, expected in zip(episodes, reference["episodes"]):
        if (actual["scenario_id"] != expected["scenario_id"] or actual["steps"] != expected["steps"] or
                not math.isclose(actual["mean_load_std"], expected["mean_load_std"], rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError("Replayed episode differs from the selected test evaluation")
        recomputed = summarize_episode(actual["step_records"])
        if any(not math.isclose(actual[name], recomputed[name], rel_tol=1e-12, abs_tol=1e-14) for name in METRICS):
            raise ValueError("Episode metrics differ from the step records")
        errors.append(abs(actual["mean_load_std"] - expected["mean_load_std"]))
    means = {name: float(np.mean([row[name] for row in episodes])) for name in METRICS}
    if not math.isclose(means["mean_load_std"], summary["selected"]["test_mean_load_std"], rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Replayed load balance differs from the selected checkpoint score")
    for name in METRICS[1:]:
        if not math.isfinite(means[name]) or not 0 <= means[name] <= 1:
            raise ValueError("Invalid association-change fraction")
    return dict(schema_version=1, scene=summary["scene"], method=summary["method"],
                history_length=summary["history_length"], seed=result["seed"],
                checkpoint=Path(checkpoint).name, checkpoint_step=summary["selected"]["step"],
                model_sha256=sha256(checkpoint), evaluation_policy="deterministic",
                num_episodes=len(episodes), episodes=episodes,
                test_environment_list_sha256=sha256(Path(env_list)), replay_max_abs_error=max(errors),
                fraction_aggregation="equal_environment_mean(time_mean(changed_users/eligible_users)); empty snapshots excluded",
                association_comparison="independent predictions with identical reports, references and hysteresis", **means)


def evaluate(args):
    set_cpu_threads(args.cpu_threads)
    summary, checkpoint, reference = selected_inputs(args.run_dir, args.seed)
    targets = collect_targets(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_checkpoint(checkpoint, targets, args.seed, device)
    result = finalize_result(result, summary, checkpoint, reference, args.env_list)
    write_json(args.output, result)
    print("mean_load_std=%.6f association_change=%.4f%% replay_error=%.3g output=%s" % (
        result["mean_load_std"], result["mean_changed_vs_previous_cio"] * 100.,
        result["replay_max_abs_error"], args.output), flush=True)
    return result


def summarize_association_changes(root):
    root = Path(root)
    rows = []
    maximum_error = 0.
    for scene, label in SCENES.items():
        for method, method_label in METHODS.items():
            runs = [read_json(root / scene / method / ("seed_%d" % seed) / "evaluation.json") for seed in SEEDS]
            expected_method = "link_mvd" if method == "link_mvd_h5" else "sac"
            expected_history = int(method.rsplit("_h", 1)[1])
            if any(run["seed"] != seed or run["scene"] != scene or
                   run["method"] != expected_method or run["history_length"] != expected_history
                   for run, seed in zip(runs, SEEDS)):
                raise ValueError("Association result is stored under the wrong scene, method or seed")
            stats = {}
            for name in METRICS:
                values = np.asarray([run[name] for run in runs], dtype=np.float64) * 100.
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite association summary")
                stats[name] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)), seed_values=values.tolist())
            rows.append(dict(setting=label, method=method_label, seeds=list(SEEDS),
                             selected_steps=[run["checkpoint_step"] for run in runs], metrics=stats))
            maximum_error = max(maximum_error, *(run["replay_max_abs_error"] for run in runs))
    summary = dict(schema_version=1, display_scale=100, sd_ddof=1, rows=rows,
                   replay_max_abs_error=maximum_error,
                   fraction_aggregation="equal_environment_mean(time_mean(changed_users/eligible_users)); empty snapshots excluded",
                   checkpoint_selection="reuse_main_comparison_training_selected_checkpoint",
                   association_comparison="independent predictions with identical reports, references and hysteresis")
    write_json(root / "summary.json", summary)
    headers = ("Setting", "Method", "Load std x100", "Updated vs previous CIO (%)",
               "Updated vs zero CIO (%)", "Updated vs reference (%)")
    table_rows = []
    for row in rows:
        values = ["%.2f +/- %.2f" % (row["metrics"][name]["mean"], row["metrics"][name]["sd"]) for name in METRICS[:4]]
        table_rows.append([row["setting"], row["method"]] + values)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table_rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir")
    mode.add_argument("--summarize", type=Path, help="Aggregate all four settings and three seeds")
    parser.add_argument("--env-list")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.run_dir and (not args.env_list or not args.output):
        parser.error("--run-dir requires --env-list and --output")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.summarize:
        summarize_association_changes(args.summarize)
    else:
        evaluate(args)
