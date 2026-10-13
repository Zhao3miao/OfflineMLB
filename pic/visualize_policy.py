"""Visualize a baseline or learned policy on one map scenario as a GIF.

Frames are captured by the environment's built-in renderer through
save_animation_path and exported as a GIF on close; a JSON summary of the
rollout is written next to it. Learned agents load a checkpoint saved by the
corresponding train_<method>.py script.
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.attention_data import observation_to_attention_inputs
from agent.clairvoyant_greedy import ClairvoyantGreedyPolicy
from env import CIOLoadBalancingEnv
from utils.experiment_utils import load_actor, make_actor_step, read_scene, write_json
from utils.public_cli import METHOD_TYPES as EXPECTED_TYPES


def make_action_fn(args, environment, device):
    if args.agent == "static":
        def select(history):
            return np.zeros(environment.action_dim, dtype=np.float32)
        return select
    if args.agent == "random":
        rng = np.random.default_rng(args.seed)

        def select(history):
            return rng.uniform(-1.0, 1.0, environment.action_dim).astype(np.float32)
        return select
    if args.agent == "greedy":
        policy = ClairvoyantGreedyPolicy(seed=args.seed)

        def select(history):
            return policy.select_action(environment)[0]
        return select
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for agent %s" % args.agent)
    actor, normalization, model_config, model_type = load_actor(Path(args.checkpoint), device)
    if model_type not in EXPECTED_TYPES[args.agent]:
        raise ValueError("Checkpoint model_type=%s does not match --agent %s" % (model_type, args.agent))
    return make_actor_step(actor, normalization, model_config, model_type, device)


def run(args):
    import torch
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    scene = read_scene(args.env)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s_seed%d" % (Path(args.env).stem, args.agent, args.seed)
    gif_path = (output_dir / (stem + ".gif")).resolve()
    summary_path = output_dir / (stem + "_summary.json")
    config = dict(scene["config"])
    if args.osm_file:
        config["osm_file_path"] = str(Path(args.osm_file).resolve())
    if args.no_map_background:
        config["osm_file_path"] = ""
    config["save_animation_path"] = str(gif_path)
    environment = CIOLoadBalancingEnv(config)
    select = make_action_fn(args, environment, device)
    try:
        observation, _ = environment.reset(seed=args.seed)
        history = deque(maxlen=64)
        history.append(observation_to_attention_inputs(environment, observation))
        rewards, load_stds, blockings, handovers = [], [], [], []
        while environment.current_time < environment.total_time - 1:
            if args.max_steps > 0 and len(rewards) >= args.max_steps:
                break
            if environment.current_time % args.render_every == 0:
                environment.render(mode="rgb_array")
            action = select(history)
            observation, reward, terminal, truncated, info = environment.step(action)
            history.append(observation_to_attention_inputs(environment, observation))
            rewards.append(float(reward))
            load_stds.append(float(-info.get("load_balance", float("nan"))))
            blockings.append(float(info.get("blocking", float("nan"))))
            handovers.append(float(info.get("handover_count", 0.0)))
            if terminal or truncated:
                break
    finally:
        environment.close()
    summary = {
        "env": str(args.env),
        "agent": args.agent,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "steps": len(rewards),
        "full_episode": environment.current_time == environment.total_time - 1,
        "return": float(sum(rewards)),
        "mean_load_std": float(np.mean(load_stds)) if load_stds else float("nan"),
        "mean_blocking": float(np.mean(blockings)) if blockings else float("nan"),
        "total_handovers": float(sum(handovers)),
        "gif": str(gif_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=str(ROOT / "data/scenarios/map/scenario_13.yaml"),
                        help="Map scenario YAML to visualize")
    parser.add_argument("--agent", choices=["static", "random", "greedy"] + list(EXPECTED_TYPES),
                        default="static")
    parser.add_argument("--checkpoint", help="Trained checkpoint for learned agents")
    parser.add_argument("--output-dir", default="pic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="Zero renders the full episode")
    parser.add_argument("--render-every", type=int, default=10, help="Capture one frame every N steps")
    parser.add_argument("--osm-file", help="Optional local OSM XML file for the road background")
    parser.add_argument("--no-map-background", action="store_true",
                        help="Skip the OSM road background for faster rendering")
    args = parser.parse_args(argv)
    if args.max_steps < 0 or args.render_every < 1:
        parser.error("max-steps must be nonnegative and render-every positive")
    if args.agent in EXPECTED_TYPES and not args.checkpoint:
        parser.error("--checkpoint is required for agent %s" % args.agent)
    return args


if __name__ == "__main__":
    run(parse_args())
