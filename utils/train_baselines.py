"""Train CQL, IQL or temporal Transformer SAC on existing offline datasets.

Individual entry points: train_cql.py, train_iql.py, train_transformer_sac.py.
Writes the same metrics/checkpoint/evaluation-compatible metadata as SAC.
Training starts from scratch; optimizer and RNG states are saved for future
resume support, but a resume CLI is deliberately not advertised here.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from agent.attention_data import load_attention_dataset, add_rsrp_normalization_args, training_normalization
from agent.cql_agent import CQLAgent
from agent.iql_agent import IQLAgent
from agent.transformer_sac_agent import TransformerSACAgent
from utils.train_history import make_loader
from utils.train_current import move_batch
from utils.experiment_utils import (
    add_common_train_args, atomic_checkpoint, checkpoint_name,
    resolve_checkpoint_steps, set_cpu_threads, write_json, add_actor_update_args,
)

AGENTS = {"cql": CQLAgent, "iql": IQLAgent, "transformer_sac": TransformerSACAgent}


def parse_args(algorithm=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if algorithm is None:
        parser.add_argument("--algorithm", choices=tuple(AGENTS), required=True)
    else:
        parser.set_defaults(algorithm=algorithm)
    add_common_train_args(parser)
    add_actor_update_args(parser)
    add_rsrp_normalization_args(parser)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    if algorithm in (None, "cql"):
        parser.add_argument("--cql-weight", type=float, default=1.0)
        parser.add_argument("--cql-temperature", type=float, default=1.0)
        parser.add_argument("--cql-num-actions", type=int, default=10)
    if algorithm in (None, "iql"):
        parser.add_argument("--expectile", type=float, default=0.7)
        parser.add_argument("--inverse-temperature", type=float, default=3.0)
        parser.add_argument("--max-weight", type=float, default=100.0)
        parser.add_argument("--value-lr", type=float, default=3e-4)
    if algorithm in (None, "transformer_sac"):
        parser.add_argument("--temporal-dim", type=int, default=128)
        parser.add_argument("--temporal-heads", type=int, default=4)
        parser.add_argument("--temporal-layers", type=int, default=1)
        parser.add_argument("--temporal-feedforward-dim", type=int, default=256)
    parser.set_defaults(train_steps=30000, checkpoint_interval=2000, actor_update_interval=2, rsrp_normalization="train_zscore", cpu_threads=2)
    args = parser.parse_args(argv)
    if args.history_length is None:
        args.history_length = 5 if args.algorithm == "transformer_sac" else 1
    if (not 1 <= args.history_length <= 64 or args.batch_size < 1 or args.log_interval < 1
            or min(args.embedding_dim, args.hidden_dim, args.num_heads) < 1
            or args.embedding_dim % args.num_heads):
        parser.error("Invalid history, batch, logging, or network dimensions")
    if args.algorithm == "iql":
        # IQL has no entropy temperature. Reject an explicitly different value.
        if args.alpha not in (0.0, 0.2):
            parser.error("IQL does not use --alpha; use --inverse-temperature for advantage weights")
        args.alpha = 0.0
    resolve_checkpoint_steps(args.train_steps, args.checkpoint_interval)
    if not 1 <= args.actor_update_interval <= args.train_steps:
        parser.error("Invalid Actor update interval")
    return args


def train(args):
    set_cpu_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    if (output / "training_metrics.jsonl").exists() or any(output.glob("*.pt")):
        raise ValueError("Output already contains training artifacts; use a new directory")
    arrays = load_attention_dataset(args.dataset)
    if len(arrays["action"]) < args.batch_size:
        raise ValueError("Dataset has fewer transitions than batch size")
    # Dataset parts may reuse trajectory names; histories must never cross scenes.
    arrays["trajectory_id"] = np.asarray([json.dumps([str(s), str(t)]) for s, t in
                                         zip(arrays["scenario_id"], arrays["trajectory_id"])])
    normalization = training_normalization(arrays, getattr(args, "rsrp_normalization", "fixed"))
    dataset, loader = make_loader(arrays, normalization, args, differential=False)
    iterator = iter(loader)
    config = dict(token_dim=int(arrays["user_obs"][0].shape[1]),
                  global_dim=int(arrays["global_obs"].shape[1]),
                  action_dim=int(arrays["action"].shape[1]), embedding_dim=args.embedding_dim,
                  hidden_dim=args.hidden_dim, num_heads=args.num_heads)
    if args.history_length > 1 or args.algorithm == "transformer_sac":
        config["history_length"] = args.history_length
    options = {key: getattr(args, key) for key in
               ("actor_lr", "critic_lr", "gamma", "tau", "reward_scale", "grad_clip")}
    options["actor_update_interval"] = getattr(args, "actor_update_interval", 1)
    if args.algorithm != "iql":
        options["alpha"] = args.alpha
    if args.algorithm == "cql":
        options.update({k: getattr(args, k) for k in ("cql_weight", "cql_temperature", "cql_num_actions")})
    elif args.algorithm == "iql":
        options.update({k: getattr(args, k) for k in
                        ("expectile", "inverse_temperature", "max_weight", "value_lr")})
    else:
        config.update({k: getattr(args, k) for k in
                       ("temporal_dim", "temporal_heads", "temporal_layers", "temporal_feedforward_dim")})
    agent = AGENTS[args.algorithm](config, device=device, **options)
    model_type = (args.algorithm if args.history_length == 1 or args.algorithm == "transformer_sac"
                  else "history_" + args.algorithm)
    metadata = dict(algorithm="sac" if args.algorithm == "transformer_sac" else args.algorithm,
                    model_type=model_type, representation=("transformer_sac" if args.algorithm == "transformer_sac"
                        else "attention_sac" if args.history_length == 1 else "history_attention_sac"),
                    history_length=args.history_length, sampling="shuffled_transitions",
                    seed=args.seed, model_config=config,
                    parameter_counts={key: sum(p.numel() for p in getattr(agent, key).parameters())
                                      for key in ("actor", "critic", "value") if hasattr(agent, key)})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "normalization.json", normalization)
    saved_steps = resolve_checkpoint_steps(args.train_steps, args.checkpoint_interval)
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.train_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            metrics = agent.update(move_batch(batch, device))
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                row = {key: float(value.cpu().item()) for key, value in metrics.items()}
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError("Non-finite metric at step %d" % step)
                row["step"] = step
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if step in saved_steps:
                state = agent.state_dict()
                state.update(metadata, normalization=normalization, train_config=vars(args), step=step,
                             rng_state=dict(python=random.getstate(), numpy=np.random.get_state(),
                                            torch=torch.get_rng_state(),
                                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                             loader_generator_state=loader.generator.get_state())
                atomic_checkpoint(output / checkpoint_name(step, args.train_steps), state)
    write_json(output / "training_complete.json", dict(
        metadata, normalization=normalization, **agent.update_counts(),
        train_steps=args.train_steps, checkpoint_interval=args.checkpoint_interval,
        saved_steps=saved_steps, num_transitions=len(dataset), device=str(device),
        dataset=str(args.dataset), train_config=vars(args)))
    return str(output / "model.pt")


if __name__ == "__main__":
    train(parse_args())
