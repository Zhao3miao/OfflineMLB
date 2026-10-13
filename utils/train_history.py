"""Train History AttentionSAC (optionally matched-exogenous differential) offline.

The differential flag keeps the matched-exogenous pair sampler and the
differential TD loss; it is retained for comparison runs but the default path
uses ordinary shuffled transitions and the standard SAC objective.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent.attention_data import load_attention_dataset, add_rsrp_normalization_args, training_normalization
from agent.attention_sac import critic_loss, make_target_critic, soft_update
from agent.history_attention_data import (
    HistoryAttentionTransitionDataset,
    MatchedExogenousBatchSampler,
    collate_history_attention_transitions,
)
from agent.history_attention_sac import (
    HistoryAttentionSACActor,
    HistoryAttentionSACCritic,
    MatchedDifferentialHistorySACActor,
    MatchedDifferentialHistorySACCritic,
    matched_differential_loss,
)
from utils.experiment_utils import add_common_train_args, add_actor_update_args, atomic_checkpoint, checkpoint_name, resolve_checkpoint_steps, set_cpu_threads, write_json
from utils.train_current import move_batch, set_requires_grad


def make_loader(arrays, normalization, args, differential):
    dataset = HistoryAttentionTransitionDataset(
        arrays, normalization, history_length=args.history_length
    )
    if differential:
        sampler = MatchedExogenousBatchSampler(
            arrays, batch_size=args.batch_size, seed=args.seed
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_history_attention_transitions,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_history_attention_transitions,
        )
    return dataset, loader


def train(args, differential=False):
    args.actor_update_interval = getattr(args, "actor_update_interval", 1)
    if not 1 <= args.actor_update_interval <= args.train_steps:
        raise ValueError("Invalid Actor update interval")
    set_cpu_threads(getattr(args, "cpu_threads", None))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    arrays = load_attention_dataset(args.dataset)
    if len(arrays["action"]) < args.batch_size:
        raise ValueError(
            "Dataset has %d transitions, fewer than batch size %d"
            % (len(arrays["action"]), args.batch_size)
        )
    # Keep histories within both a scene and a trajectory, even if labels repeat.
    arrays["trajectory_id"] = np.asarray([
        json.dumps([str(scene), str(trajectory)])
        for scene, trajectory in zip(arrays["scenario_id"], arrays["trajectory_id"])
    ])
    normalization = training_normalization(arrays, getattr(args, "rsrp_normalization", "fixed"))
    dataset, loader = make_loader(arrays, normalization, args, differential)
    iterator = iter(loader)

    token_dim = int(arrays["user_obs"][0].shape[1])
    global_dim = int(arrays["global_obs"].shape[1])
    action_dim = int(arrays["action"].shape[1])
    model_type = "matched_differential_history_sac" if differential else "history_attention_sac"
    actor_class = MatchedDifferentialHistorySACActor if differential else HistoryAttentionSACActor
    critic_class = MatchedDifferentialHistorySACCritic if differential else HistoryAttentionSACCritic
    model_config = {
        "token_dim": token_dim,
        "global_dim": global_dim,
        "action_dim": action_dim,
        "history_length": args.history_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
    }
    actor = actor_class(**model_config).to(device)
    critic = critic_class(**model_config).to(device)
    target_critic = make_target_critic(critic).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    output_dir = Path(args.output_dir)
    if (output_dir / "training_metrics.jsonl").exists() or any(output_dir.glob("*.pt")):
        raise ValueError("Output already contains training artifacts; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "normalization.json", normalization)
    checkpoint_steps = set(resolve_checkpoint_steps(args.train_steps, args.checkpoint_interval))
    metrics_path = output_dir / "training_metrics.jsonl"
    actor_updates = 0

    with metrics_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.train_steps + 1):
            record = step == 1 or step % args.log_interval == 0 or step == args.train_steps
            actor_updated = step % args.actor_update_interval == 0
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)

            with torch.no_grad():
                next_action, next_log_prob = actor.sample(
                    batch["next_user_history"],
                    batch["next_user_mask"],
                    batch["next_global_history"],
                    batch["next_history_mask"],
                )
                next_q1, next_q2 = target_critic(
                    batch["next_user_history"],
                    batch["next_user_mask"],
                    batch["next_global_history"],
                    batch["next_history_mask"],
                    next_action,
                )
                next_value = torch.min(next_q1, next_q2) - args.alpha * next_log_prob
                target_q = args.reward_scale * batch["reward"] + (
                    args.gamma * (1.0 - batch["terminal"]) * next_value
                )

            q1, q2 = critic(
                batch["user_history"],
                batch["user_mask"],
                batch["global_history"],
                batch["history_mask"],
                batch["action"],
            )
            if differential:
                q_loss, absolute_loss, differential_loss = matched_differential_loss(
                    q1, q2, target_q, args.differential_weight
                )
            else:
                q_loss = critic_loss(q1, q2, target_q)
                absolute_loss = q_loss.detach()
                differential_loss = torch.zeros((), device=device)
            critic_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            critic_optimizer.step()

            if actor_updated or record:
                set_requires_grad(critic, False)
                with torch.set_grad_enabled(actor_updated):
                    sampled_action, log_prob = actor.sample(
                        batch["user_history"], batch["user_mask"],
                        batch["global_history"], batch["history_mask"],
                    )
                    actor_q1, actor_q2 = critic(
                        batch["user_history"], batch["user_mask"],
                        batch["global_history"], batch["history_mask"], sampled_action,
                    )
                    policy_loss = (args.alpha * log_prob - torch.min(actor_q1, actor_q2)).mean()
                if actor_updated:
                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    actor_updates += 1
                set_requires_grad(critic, True)
            soft_update(critic, target_critic, args.tau)

            if record:
                row = {
                    "step": step,
                    "actor_updated": actor_updated,
                    "actor_updates": actor_updates,
                    "critic_updates": step,
                    "critic_loss": float(q_loss.detach().cpu().item()),
                    "absolute_td_loss": float(absolute_loss.detach().cpu().item()),
                    "differential_td_loss": float(differential_loss.detach().cpu().item()),
                    "actor_loss": float(policy_loss.detach().cpu().item()),
                    "q_mean": float(torch.min(q1, q2).detach().mean().cpu().item()),
                    "target_q_mean": float(target_q.detach().mean().cpu().item()),
                }
                if not all(np.isfinite(value) for key, value in row.items() if key != "step"):
                    raise FloatingPointError("Non-finite training metric at step %d" % step)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)

            if step in checkpoint_steps:
                checkpoint = {
                    "model_type": model_type,
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "target_critic": target_critic.state_dict(),
                    "normalization": normalization,
                    "model_config": model_config,
                    "train_config": vars(args),
                    "step": step,
                    "actor_update_interval": args.actor_update_interval,
                    "actor_updates": actor_updates,
                    "critic_updates": step,
                    "target_update_interval": 1,
                    "target_updates": step,
                }
                atomic_checkpoint(output_dir / checkpoint_name(step, args.train_steps), checkpoint)

    write_json(
        output_dir / "training_complete.json",
        {
            "seed": args.seed,
            "model_type": model_type,
            "train_steps": args.train_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "saved_steps": sorted(checkpoint_steps),
            "num_transitions": len(dataset),
            "device": str(device),
            "dataset": str(args.dataset),
            "differential_weight": args.differential_weight if differential else None,
            "model_config": model_config,
            "normalization": normalization,
            "actor_update_interval": args.actor_update_interval,
            "actor_updates": actor_updates,
            "critic_updates": args.train_steps,
            "target_update_interval": 1,
            "target_updates": args.train_steps,
        },
    )
    print("Saved model: %s" % (output_dir / "model.pt"), flush=True)
    return str(output_dir / "model.pt")


def make_parser(differential=False):
    model_name = (
        "matched-exogenous differential History SAC"
        if differential
        else "History AttentionSAC"
    )
    parser = argparse.ArgumentParser(
        description="Train %s on an offline attention dataset." % model_name
    )
    add_common_train_args(parser)
    add_actor_update_args(parser)
    add_rsrp_normalization_args(parser)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--differential-weight", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    train(make_parser(differential=False).parse_args(), differential=False)
