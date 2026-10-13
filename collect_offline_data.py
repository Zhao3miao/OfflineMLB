"""Collect random-action offline trajectories from prepared snapshot environments."""
import argparse
from pathlib import Path

import numpy as np

from agent.attention_data import DATASET_SCHEMA_VERSION, make_fixed_normalization
from utils.collection import append_transition, empty_storage, save_dataset
from utils.experiment_utils import collect_targets, load_environment, sha256, write_json


def collect(args):
    targets = collect_targets(args)
    output = Path(args.output_dir)
    if (output / "train_manifest.json").exists() or any(output.glob("*.npz")):
        raise FileExistsError("Choose a fresh collection directory")
    output.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(args.seed).spawn(len(targets) * args.rollouts)
    map_rng = np.random.RandomState(args.seed)
    parts, hashes = [], {}
    for index, target in enumerate(targets):
        environment = load_environment(target)
        try:
            if environment.max_active_users > args.user_count_cap:
                raise ValueError("user-count-cap is smaller than the largest snapshot")
            normalization = make_fixed_normalization(environment.num_base_stations,
                rsrp_min=-157, rsrp_max=-30, user_count_cap=args.user_count_cap)
            storage = empty_storage()
            for repeat in range(args.rollouts):
                episode = index * args.rollouts + repeat
                rng = map_rng if target["kind"] == "map" else np.random.default_rng(streams[episode])
                observation, _ = environment.reset(seed=args.seed + episode)
                for step in range(args.max_steps or environment.control_steps):
                    action = rng.uniform(environment.action_space.low, environment.action_space.high).astype(np.float32)
                    following, reward, terminal, truncated, info = environment.step(action)
                    done = terminal or truncated or (args.max_steps and step + 1 == args.max_steps)
                    append_transition(storage, environment, observation, following, action, reward,
                        done, info, environment.get_current_info(), target["scenario_id"], step, "random",
                        trajectory_id="%s:rollout_%d" % (target["scenario_id"], repeat))
                    observation = following
                    if done:
                        break
            part = target["scenario_id"] + ".npz"
            save_dataset(output / part, storage, normalization)
            parts.append(part)
            hashes[part] = sha256(output / part)
            write_json((output / part).with_suffix(".json"), dict(scenario_id=target["scenario_id"],
                data_seed=args.seed, trajectories_per_environment=args.rollouts,
                num_transitions=len(storage["reward"]), sha256=hashes[part], policy="random",
                dataset_schema_version=DATASET_SCHEMA_VERSION))
        finally:
            environment.close()
        print("Collected %s" % target["scenario_id"], flush=True)
    write_json(output / "train_manifest.json", dict(format="attention_dataset_parts_v1", parts=parts,
        part_sha256=hashes, trajectories_per_environment=args.rollouts, data_seed=args.seed,
        training_environments=[target["scenario_id"] for target in targets]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--user-count-cap", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.rollouts < 1 or args.max_steps < 0:
        parser.error("rollouts must be positive and max-steps nonnegative")
    collect(args)
