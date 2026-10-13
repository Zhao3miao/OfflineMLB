"""Train AttentionSAC (single-snapshot set encoder) on an offline attention dataset.

Writes the final model as model.pt plus optional interval checkpoints, a
training metrics log and training_complete.json into --output-dir. Resuming
is the caller's responsibility; this script always trains from scratch.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent.attention_data import (
    AttentionTransitionDataset,
    collate_attention_transitions,
    load_attention_dataset,
    add_rsrp_normalization_args, training_normalization,
)
from agent.attention_sac import (
    AttentionSACActor,
    AttentionSACCritic,
    critic_loss,
    make_target_critic,
    soft_update,
)
from utils.experiment_utils import add_actor_update_args, add_common_train_args, atomic_checkpoint, checkpoint_name, resolve_checkpoint_steps, set_cpu_threads, write_json


def move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train(args):
    args.actor_update_interval = getattr(args, "actor_update_interval", 1)
    if not 1 <= args.actor_update_interval <= args.train_steps:
        raise ValueError("actor-update-interval must be between 1 and train-steps")
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
    normalization = training_normalization(arrays, getattr(args, "rsrp_normalization", "fixed"))
    dataset = AttentionTransitionDataset(arrays, normalization)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_attention_transitions,
    )
    iterator = iter(loader)

    token_dim = int(arrays["user_obs"][0].shape[1])
    global_dim = int(arrays["global_obs"].shape[1])
    action_dim = int(arrays["action"].shape[1])
    model_config = {
        "token_dim": token_dim,
        "global_dim": global_dim,
        "action_dim": action_dim,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
    }
    actor = AttentionSACActor(**model_config).to(device)
    critic = AttentionSACCritic(**model_config).to(device)
    target_critic = make_target_critic(critic).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    output_dir = Path(args.output_dir)
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
                    batch["next_user_obs"],
                    batch["next_user_mask"],
                    batch["next_global_obs"],
                )
                next_q1, next_q2 = target_critic(
                    batch["next_user_obs"],
                    batch["next_user_mask"],
                    batch["next_global_obs"],
                    next_action,
                )
                next_value = torch.min(next_q1, next_q2) - args.alpha * next_log_prob
                target_q = args.reward_scale * batch["reward"] + (
                    args.gamma * (1.0 - batch["terminal"]) * next_value
                )

            q1, q2 = critic(
                batch["user_obs"], batch["user_mask"], batch["global_obs"], batch["action"]
            )
            q_loss = critic_loss(q1, q2, target_q)
            critic_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            critic_optimizer.step()

            if actor_updated or record:
                set_requires_grad(critic, False)
                # Logged skipped steps measure the current loss without updating.
                with torch.set_grad_enabled(actor_updated):
                    sampled_action, log_prob = actor.sample(
                        batch["user_obs"], batch["user_mask"], batch["global_obs"]
                    )
                    actor_q1, actor_q2 = critic(
                        batch["user_obs"], batch["user_mask"], batch["global_obs"], sampled_action,
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
                    "model_type": "attention_sac",
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
            "model_type": "attention_sac",
            "train_steps": args.train_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "saved_steps": sorted(checkpoint_steps),
            "num_transitions": len(dataset),
            "device": str(device),
            "dataset": str(args.dataset),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train offline AttentionSAC.")
    add_common_train_args(parser)
    add_actor_update_args(parser)
    add_rsrp_normalization_args(parser)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
