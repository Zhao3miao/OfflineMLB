"""Shared command-line handling for the separate paper-method entry points."""
import argparse
import math
from pathlib import Path
import re
import shutil

import torch

from agent.attention_data import add_rsrp_normalization_args
from agent.user_attention_link_sac import REPRESENTATION
from utils.experiment_utils import (
    add_actor_update_args, add_common_train_args, collect_targets, evaluate_model,
    read_json, set_cpu_threads, sha256, write_json,
)


def training_args(method, argv=None):
    parser = argparse.ArgumentParser(description="Train %s from offline data." % method)
    add_common_train_args(parser)
    add_actor_update_args(parser)
    add_rsrp_normalization_args(parser)
    parser.set_defaults(train_steps=30000, checkpoint_interval=2000,
                        actor_update_interval=2, rsrp_normalization="train_zscore", cpu_threads=2)
    link = method.startswith("link")
    parser.add_argument("--history-length", type=int, default=5 if link else 1)
    parser.add_argument("--sampling", choices=("shuffled", "matched"), default="matched" if method.endswith("mvd") else "shuffled")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128 if link else 256)
    parser.add_argument("--summary-dim", type=int, default=32)
    parser.add_argument("--temporal-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=2)
    args = parser.parse_args(argv)
    if method.endswith("mvd") and args.sampling != "matched":
        parser.error("MVD requires matched sampling")
    args.differential_weight = 1.0 if method.endswith("mvd") else 0.0
    args.representation = REPRESENTATION if link else "attention_sac" if args.history_length == 1 else "history_attention_sac"
    args.shared_target_noise = args.detach_differential_encoder = False
    return args


def train_method(method, argv=None):
    args = training_args(method, argv)
    output = Path(args.output_dir)
    if any(output.glob("*.pt")) or (output / "training_metrics.jsonl").exists():
        raise ValueError("Training already exists; choose a new output directory")
    # Retain each original training path and its RNG/update ordering.
    if method == "sac" and args.sampling == "shuffled":
        if args.history_length == 1:
            from utils.train_current import train
            return train(args)
        from utils.train_history import train
        return train(args, differential=False)
    from utils.train_matched import train
    return train(args, use_matched=args.sampling == "matched")


METHOD_TYPES = {
    "sac": ("attention_sac", "history_attention_sac"),
    "sac_mvd": ("attention_sac", "history_attention_sac", "matched_differential_history_sac"),
    "link": (REPRESENTATION,), "link_mvd": (REPRESENTATION,),
    "cql": ("cql", "history_cql"), "iql": ("iql", "history_iql"),
    "transformer_sac": ("transformer_sac",),
}


def summarize_run(directory, method, seed):
    """Select by training scores and retain a portable selected checkpoint."""
    directory = Path(directory)
    previous_path = directory / "checkpoint_summary.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    metadata_path = directory / "training_complete.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    records = []
    for path in sorted((directory / "evaluations").glob("step_*_train.json")):
        step = int(path.name.split("_")[1])
        train = read_json(path)
        test = read_json(path.with_name("step_%d_test.json" % step))
        if train["seed"] != seed or test["seed"] != seed:
            raise ValueError("Evaluation seed does not match the run")
        records.append(dict(step=step, train_mean_load_std=train["mean_load_std"],
                            test_mean_load_std=test["mean_load_std"]))
    records.sort(key=lambda row: row["step"])
    if not records or any(not math.isfinite(row[key]) for row in records
                          for key in ("train_mean_load_std", "test_mean_load_std")):
        raise ValueError("Missing or non-finite checkpoint evaluations")
    if metadata.get("saved_steps") and [row["step"] for row in records] != metadata["saved_steps"]:
        raise ValueError("Evaluations do not cover the saved checkpoint steps")
    selected = min(records, key=lambda row: (row["train_mean_load_std"], row["step"]))
    from utils.experiment_utils import checkpoint_name
    source = directory / checkpoint_name(selected["step"], metadata.get("train_steps", records[-1]["step"]))
    destination = directory / "selected.pt"
    if source.exists():
        temporary = directory / "selected.pt.tmp"
        shutil.copyfile(str(source), str(temporary))
        temporary.replace(destination)
    elif not destination.exists() or previous.get("selected", {}).get("step") != selected["step"]:
        raise ValueError("The checkpoint selected by training scores is unavailable")
    scene = next((part for part in reversed(directory.parts)
                  if part in ("map", "site_a", "site_b", "site_c")), None)
    history = re.search(r"_h(\d+)(?:_|$)", directory.parent.name)
    rollouts = re.search(r"rollouts_(\d+)", str(directory) + "/" + metadata.get("dataset", ""))
    summary = dict(scene=scene, method=method, seed=seed,
                   history_length=int(history.group(1)) if history else metadata.get("model_config", {}).get("history_length", previous.get("history_length", 1)),
                   rollouts=int(rollouts.group(1)) if rollouts else previous.get("rollouts", 64),
                   checkpoints=records, selected=selected, selected_checkpoint="selected.pt",
                   selection_metric="min_train_mean_load_std", model_sha256=sha256(destination))
    write_json(previous_path, summary)
    return summary


def evaluate_method(method, argv=None):
    parser = argparse.ArgumentParser(description="Evaluate %s and select checkpoints on training environments." % method)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--run-dir", help="Evaluate every saved checkpoint, then select by training score")
    parser.add_argument("--env-list", help="Environment list for a single checkpoint")
    parser.add_argument("--train-env-list")
    parser.add_argument("--test-env-list")
    parser.add_argument("--output", help="JSON output for a single checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild selection metadata from saved evaluations without replaying models")
    args = parser.parse_args(argv)
    if args.max_steps < 0:
        parser.error("max-steps must be nonnegative")
    if args.summarize_only:
        if not args.run_dir:
            parser.error("--summarize-only requires --run-dir")
        summary = summarize_run(args.run_dir, method, args.seed)
        print("mean_load_std=%.6f" % summary["selected"]["test_mean_load_std"], flush=True)
        return summary
    set_cpu_threads(args.cpu_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def score(path, env_list, output):
        targets = collect_targets(argparse.Namespace(env=None, env_list=env_list))
        result = evaluate_model(path, targets, device, args.seed, args.max_steps, METHOD_TYPES[method])
        write_json(output, result)
        return result["mean_load_std"]

    if args.checkpoint:
        if not args.env_list or not args.output:
            parser.error("--checkpoint requires --env-list and --output")
        value = score(Path(args.checkpoint), args.env_list, args.output)
    else:
        if not args.train_env_list or not args.test_env_list:
            parser.error("--run-dir requires --train-env-list and --test-env-list")
        directory = Path(args.run_dir)
        metadata = read_json(directory / "training_complete.json")
        from utils.experiment_utils import checkpoint_name
        records = []
        for step in metadata["saved_steps"]:
            path = directory / checkpoint_name(step, metadata["train_steps"])
            train = score(path, args.train_env_list, directory / "evaluations" / ("step_%d_train.json" % step))
            test = score(path, args.test_env_list, directory / "evaluations" / ("step_%d_test.json" % step))
            records.append(dict(step=step, train_mean_load_std=train, test_mean_load_std=test))
        summary = summarize_run(directory, method, args.seed)
        value = summary["selected"]["test_mean_load_std"]
    print("mean_load_std=%.6f" % value, flush=True)
