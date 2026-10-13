"""Shared helpers for offline train/eval scripts and experiment preparation.

Keeps path handling, locking, checksums, environment discovery and unified
evaluation output in one place so train_<method>.py and eval_<method>.py stay
thin. Environment targets are auto-detected: a YAML file or a directory of
YAML files is a map scenario, a directory containing manifest.json and
snapshots.npz is a processed MRO window.
"""
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from agent.attention_data import normalize_single, observation_to_attention_inputs
from agent.attention_sac import AttentionSACActor
from agent.history_attention_data import normalize_history
from agent.history_attention_sac import (
    HistoryAttentionSACActor,
    MatchedDifferentialHistorySACActor,
)
from agent.user_attention_link_sac import REPRESENTATION as USER_ATTENTION_REPRESENTATION, UserAttentionLinkActor, MEAN_ATTENTION_POOLING
from agent.transformer_sac_agent import TransformerSACActor
from env import CIOLoadBalancingEnv, RealDataCIOLoadBalancingEnv

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ("steady", "tide", "decay", "surge")

ACTOR_CLASSES = {
    "attention_sac": AttentionSACActor, "history_attention_sac": HistoryAttentionSACActor,
    "matched_differential_history_sac": MatchedDifferentialHistorySACActor,
    "cql": AttentionSACActor, "history_cql": HistoryAttentionSACActor,
    "iql": AttentionSACActor, "history_iql": HistoryAttentionSACActor,
    "transformer_sac": TransformerSACActor, USER_ATTENTION_REPRESENTATION: UserAttentionLinkActor,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def file_lock(path):
    """OS-level exclusive lock released on crash; never infer ownership from PID files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_status(directory, stage, **extra):
    write_json(Path(directory) / "status.json", {"pid": os.getpid(), "stage": stage,
               "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, str(temporary))
    temporary.replace(path)
    return path


def resolve_checkpoint_steps(train_steps, checkpoint_interval):
    """Steps at which a checkpoint is written; final step is always included."""
    interval = train_steps if checkpoint_interval is None else checkpoint_interval
    if interval < 1 or train_steps < 1:
        raise ValueError("train_steps and checkpoint_interval must be positive")
    steps = list(range(interval, train_steps, interval))
    if train_steps not in steps:
        steps.append(train_steps)
    return steps


def checkpoint_name(step, train_steps):
    return "model.pt" if step == train_steps else "checkpoint_%d.pt" % step


def read_scene(path):
    """Load one map scenario YAML into the portable scene dict used by eval."""
    path = Path(path)
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return {"path": str(path), "sha256": sha256(path), "scenario_id": config["scenario_id"], "config": config}


def expand_target(path):
    """Expand one --env argument into a list of environment targets."""
    path = Path(path)
    if path.is_dir():
        if (path / "manifest.json").is_file() and (path / "snapshots.npz").is_file():
            return [make_mro_target(path)]
        return [make_map_target(p) for p in sorted(path.glob("*.yaml"))]
    if path.suffix == ".yaml":
        return [make_map_target(path)]
    if path.suffix == ".json":
        value = read_json(path)
        paths = value if isinstance(value, list) else value.get("paths", [])
        if not paths:
            raise ValueError("JSON target list must be a list or contain a paths field: %s" % path)
        return [target for item in paths for target in expand_target(Path(item) if Path(item).is_absolute() or Path(item).exists() else path.parent / item)]
    raise ValueError("Unsupported environment target: %s" % path)


def collect_targets(args):
    """Collect environment targets from --env and --env-list arguments."""
    if not getattr(args, "env", None) and not getattr(args, "env_list", None):
        raise ValueError("Provide at least one of --env or --env-list")
    targets = []
    if getattr(args, "env", None):
        targets.extend(expand_target(args.env))
    if getattr(args, "env_list", None):
        targets.extend(expand_target(args.env_list))
    if not targets:
        raise ValueError("No environment targets found")
    return targets


def make_map_target(path):
    scene = read_scene(path)
    # Road environments are no longer grouped by population profile; older
    # scenario files still carry one, so the field keeps the profile when present
    # and otherwise reports the shared map kind.
    population = scene["config"].get("population_config") or {}
    return {"path": scene["path"], "kind": "map", "scenario_id": scene["scenario_id"],
            "profile": population.get("profile", "map"),
            "sha256": scene["sha256"], "config": scene["config"]}


def make_mro_target(directory):
    manifest = read_json(directory / "manifest.json")
    return {"path": str(directory), "kind": "mro",
            "scenario_id": manifest["scenario_id"], "profile": "mro", "config": None}


def load_environment(target):
    return (RealDataCIOLoadBalancingEnv(target["path"]) if target["kind"] == "mro"
            else CIOLoadBalancingEnv(target["config"]))


def load_actor(checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type not in ACTOR_CLASSES:
        raise ValueError("Unsupported model type: %s" % model_type)
    if model_type == USER_ATTENTION_REPRESENTATION and checkpoint["model_config"].get("link_pooling", MEAN_ATTENTION_POOLING) != MEAN_ATTENTION_POOLING:
        raise ValueError("Checkpoint must use the full mean-plus-attention Link representation")
    actor = ACTOR_CLASSES[model_type](**checkpoint["model_config"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, checkpoint["normalization"], checkpoint["model_config"], model_type


def make_actor_step(actor, normalization, model_config, model_type, device):
    """Return an action-selection callable observing the model's input style."""
    history_length = int(model_config.get("history_length", 1))
    uses_history = model_type not in ("attention_sac", "cql", "iql")

    def select(observation_history):
        if uses_history:
            inputs = normalize_history(observation_history, normalization, history_length)
        else:
            inputs = normalize_single(*observation_history[-1], normalization)
        with torch.no_grad():
            action, _ = actor.sample(*(value.to(device) for value in inputs), deterministic=True)
        return action.cpu().numpy()[0]

    return select


def run_episode(environment, select_action, seed, max_steps=0):
    """Roll out one episode; returns per-episode metrics and raw step data."""
    observation, _ = environment.reset(seed=seed)
    history = deque(maxlen=64)
    history.append(observation_to_attention_inputs(environment, observation))
    rewards, load_stds, blockings, handovers, steps = [], [], [], [], 0
    done = False
    while not done:
        if max_steps > 0 and steps >= max_steps:
            break
        action = select_action(history)
        observation, reward, terminal, truncated, info = environment.step(action)
        history.append(observation_to_attention_inputs(environment, observation))
        rewards.append(float(reward))
        load_stds.append(float(-info.get("load_balance", float("nan"))))
        blockings.append(float(info.get("blocking", float("nan"))))
        handovers.append(float(info.get("handover_count", 0.0)))
        steps += 1
        done = terminal or truncated
    return {"steps": steps, "return": float(sum(rewards)),
            "mean_load_std": float(np.mean(load_stds)) if load_stds else float("nan"),
            "mean_blocking": float(np.mean(blockings)) if blockings else float("nan"),
            "total_handovers": float(sum(handovers))}


def evaluate_model(checkpoint_path, targets, device, seed, max_steps=0, expected_types=None):
    actor, normalization, model_config, model_type = load_actor(checkpoint_path, device)
    if expected_types is not None and model_type not in expected_types:
        raise ValueError("Checkpoint model_type=%s not in %s" % (model_type, expected_types))
    select = make_actor_step(actor, normalization, model_config, model_type, device)
    episodes = []
    for index, target in enumerate(targets):
        environment = load_environment(target)
        try:
            row = run_episode(environment, select, seed + index, max_steps)
            row.update({"scenario_id": target["scenario_id"], "profile": target["profile"], "kind": target["kind"]})
            episodes.append(row)
        finally:
            environment.close()
    values = [episode["mean_load_std"] for episode in episodes if np.isfinite(episode["mean_load_std"])]
    return {"checkpoint": str(checkpoint_path), "seed": seed, "model_type": model_type,
            "model_config": model_config, "normalization": normalization, "episodes": episodes,
            "mean_load_std": float(np.mean(values)) if values else float("nan"),
            "std_across_scenarios": float(np.std(values)) if len(values) > 1 else 0.0,
            "evaluation_policy": "deterministic", "num_episodes": len(episodes)}


def baseline_step(policy, environment, seed):
    rng = np.random.default_rng(seed)

    def select(history):
        if policy == "static_cio":
            return np.zeros(environment.action_dim, dtype=np.float32)
        return rng.uniform(-1.0, 1.0, environment.action_dim).astype(np.float32)

    return select


def evaluate_baseline(targets, policy, seed, max_steps=0):
    if policy not in ("static_cio", "random"):
        raise ValueError("Unsupported baseline policy: %s" % policy)
    episodes = []
    for index, target in enumerate(targets):
        environment = load_environment(target)
        try:
            step_seed = seed + index
            select = baseline_step(policy, environment, step_seed)
            row = run_episode(environment, select, step_seed, max_steps)
            row.update({"scenario_id": target["scenario_id"], "profile": target["profile"], "kind": target["kind"]})
            episodes.append(row)
        finally:
            environment.close()
    values = [episode["mean_load_std"] for episode in episodes if np.isfinite(episode["mean_load_std"])]
    return {"policy": policy, "seed": seed, "episodes": episodes,
            "mean_load_std": float(np.mean(values)) if values else float("nan"),
            "std_across_scenarios": float(np.std(values)) if len(values) > 1 else 0.0,
            "evaluation_policy": policy, "num_episodes": len(episodes)}


def set_cpu_threads(cpu_threads):
    if cpu_threads is not None and cpu_threads >= 1:
        torch.set_num_threads(int(cpu_threads))


def add_common_eval_args(parser):
    parser.add_argument("--env", help="Map YAML file, directory of YAML files, processed MRO directory or JSON list")
    parser.add_argument("--env-list", help="JSON file with a list of environment paths")
    parser.add_argument("--output", required=True, help="Result JSON output path")
    parser.add_argument("--max-steps", type=int, default=0, help="Zero evaluates full episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)


def add_actor_update_args(parser):
    parser.add_argument("--actor-update-interval", type=int, default=1,
                        help="Critic updates per Actor update; target Critic still updates every step")


def add_common_train_args(parser):
    parser.add_argument("--dataset", required=True, help="Attention dataset NPZ or manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Torch intra-op thread count; default keeps torch default")
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Save a checkpoint every N steps; default saves only the final model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=100)
