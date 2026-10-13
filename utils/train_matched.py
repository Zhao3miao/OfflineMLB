"""Shared TD and matched value-difference training for SAC and Link."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from agent.attention_data import load_attention_dataset, training_normalization
from agent.attention_sac import AttentionSACActor, AttentionSACCritic, critic_loss, make_target_critic, soft_update
from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic, matched_differential_loss
from agent.user_attention_link_sac import REPRESENTATION as USER_ATTENTION_REPRESENTATION, UserAttentionLinkActor, UserAttentionLinkCritic, MEAN_ATTENTION_POOLING
from utils.train_current import move_batch, set_requires_grad
from utils.train_history import make_loader
from utils.experiment_utils import atomic_checkpoint, checkpoint_name, resolve_checkpoint_steps, set_cpu_threads, write_json
ALGORITHM = "matched_differential_sac"
REPRESENTATIONS = {"attention_sac": (AttentionSACActor, AttentionSACCritic),
                  "history_attention_sac": (HistoryAttentionSACActor, HistoryAttentionSACCritic),
                  USER_ATTENTION_REPRESENTATION: (UserAttentionLinkActor, UserAttentionLinkCritic)}
ATTENTION_REPRESENTATIONS = LINK_REPRESENTATIONS = DIAGNOSTIC_REPRESENTATIONS = (USER_ATTENTION_REPRESENTATION,)
def resolve_args(args):
    args.actor_update_interval = getattr(args, "actor_update_interval", 1)
    if not 1 <= args.actor_update_interval <= args.train_steps:
        raise ValueError("actor-update-interval must be between 1 and train-steps")
    args.shared_target_noise = getattr(args, "shared_target_noise", False)
    args.detach_differential_encoder = getattr(args, "detach_differential_encoder", False)
    if args.detach_differential_encoder:
        raise ValueError("detach-differential-encoder currently requires link_snapshot_sac")
    if args.representation not in REPRESENTATIONS:
        raise ValueError("Unsupported representation: %s" % args.representation)
    if args.history_length is None:
        args.history_length = 1 if args.representation == "attention_sac" else 5
    if args.hidden_dim is None:
        args.hidden_dim = 128 if args.representation in LINK_REPRESENTATIONS else 256
    if args.representation == "attention_sac" and args.history_length != 1:
        raise ValueError("attention_sac requires H=1; use history_attention_sac for history")
    # The shared online evaluator retains at most 64 observations.
    if not 1 <= args.history_length <= 64:
        raise ValueError("history-length must lie in [1, 64]")
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("Matched sampling requires a positive even batch-size")
    if not np.isfinite(args.differential_weight) or args.differential_weight < 0:
        raise ValueError("differential-weight must be finite and nonnegative")
    if args.train_steps <= 0 or args.log_interval <= 0:
        raise ValueError("train-steps and log-interval must be positive")
    if args.representation in ATTENTION_REPRESENTATIONS:
        heads = getattr(args, "attention_heads", 2)
        if min(args.summary_dim, args.temporal_dim, args.hidden_dim, heads) < 1 or args.summary_dim % heads:
            raise ValueError("Positive dimensions required; summary-dim must be divisible by attention-heads")
    resolve_checkpoint_steps(args.train_steps, args.checkpoint_interval)
    return args


def build_models(arrays, args):
    config = {
        "token_dim": int(arrays["user_obs"][0].shape[1]),
        "global_dim": int(arrays["global_obs"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "hidden_dim": args.hidden_dim,
    }
    if args.representation in LINK_REPRESENTATIONS:
        config.update(history_length=args.history_length, summary_dim=args.summary_dim,
                      temporal_dim=args.temporal_dim)
        if args.representation in ATTENTION_REPRESENTATIONS:
            config.update(fusion="residual", attention_heads=getattr(args, "attention_heads", 2))
    else:
        config.update(embedding_dim=args.embedding_dim, num_heads=args.num_heads)
        if args.representation == "history_attention_sac":
            config["history_length"] = args.history_length
    actor_class, critic_class = REPRESENTATIONS[args.representation]
    return actor_class(**config), critic_class(**config), config


def model_inputs(batch, representation, following=False):
    prefix = "next_" if following else ""
    inputs = [batch[prefix + key] for key in
              ("user_history", "user_mask", "global_history", "history_mask")]
    if representation == "attention_sac":
        return [value[:, -1] for value in inputs[:3]]
    return inputs


def train(args, use_matched=True):
    args = resolve_args(args)
    if args.detach_differential_encoder and not use_matched:
        raise ValueError("detach-differential-encoder requires matched training")
    if args.shared_target_noise and not use_matched:
        raise ValueError("shared-target-noise requires matched training")
    set_cpu_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    branch_data = False
    arrays = load_attention_dataset(args.dataset)
    if len(arrays["action"]) < args.batch_size:
        raise ValueError("Dataset has fewer transitions than batch-size")
    # Separate scenes can reuse trajectory labels such as rollout_0. Keep
    # history windows within a scene as well as within a trajectory.
    if not branch_data:
        arrays["trajectory_id"] = np.asarray([
            json.dumps([str(scene), str(trajectory)])
            for scene, trajectory in zip(arrays["scenario_id"], arrays["trajectory_id"])
        ])
    normalization = training_normalization(arrays, getattr(args, "rsrp_normalization", "fixed"))
    dataset, loader = make_loader(
        arrays, normalization, args, differential=use_matched)
    iterator = iter(loader)
    actor, critic, model_config = build_models(arrays, args)
    actor, critic = actor.to(device), critic.to(device)
    target_critic = make_target_critic(critic).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "normalization.json", normalization)
    checkpoint_steps = set(resolve_checkpoint_steps(args.train_steps, args.checkpoint_interval))
    metadata = {
        "actor_update_interval": args.actor_update_interval,
        "target_update_interval": 1,
        "normalization": normalization,
        "algorithm": ALGORITHM if use_matched else "sac",
        # Keep the architecture identifier recognized by existing evaluators.
        "model_type": args.representation,
        "representation": args.representation,
        "differential_weight": args.differential_weight if use_matched else 0.0,
        "sampling": (("matched_same_state_distinct_actions" if branch_data else
                      "matched_scenario_time_distinct_trajectories") if use_matched else "shuffled_transitions"),
        "data_semantics": "same_state_branches_v1" if branch_data else "parallel_trajectories",
        "matched_group_count": len(loader.batch_sampler.groups) if use_matched else 0,
        "history_length": args.history_length,
        "differential_gradient_scope": (
            "q_heads_only" if args.detach_differential_encoder else
            "encoder_and_q_heads" if use_matched else "disabled"),
        "target_action_noise": "shared_within_matched_pair" if args.shared_target_noise else "independent",
    }
    if args.representation in ATTENTION_REPRESENTATIONS:
        metadata.update(link_pooling=MEAN_ATTENTION_POOLING)
    actor_updates = 0
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.train_steps + 1):
            record = step == 1 or step % args.log_interval == 0 or step == args.train_steps
            actor_updated = step % args.actor_update_interval == 0
            if args.representation in DIAGNOSTIC_REPRESENTATIONS:
                actor.history_encoder.record_diagnostics = record
                critic.history_encoder.record_diagnostics = record
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            current = model_inputs(batch, args.representation)
            following = model_inputs(batch, args.representation, following=True)
            with torch.no_grad():
                next_action, next_log_prob = actor.sample(*following)
                next_q1, next_q2 = target_critic(*following, next_action)
                next_value = torch.min(next_q1, next_q2) - args.alpha * next_log_prob
                target_q = args.reward_scale * batch["reward"] + args.gamma * (
                    1.0 - batch["terminal"]) * next_value

            q1, q2 = critic(*current, batch["action"])
            if use_matched:
                loss, absolute, differential = matched_differential_loss(q1, q2, target_q, args.differential_weight)
            else:
                loss = absolute = critic_loss(q1, q2, target_q)
                differential = loss.new_zeros(())
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            critic_optimizer.step()

            if actor_updated or record:
                set_requires_grad(critic, False)
                with torch.set_grad_enabled(actor_updated):
                    action, log_prob = actor.sample(*current)
                    actor_q1, actor_q2 = critic(*current, action)
                    actor_loss = (args.alpha * log_prob - torch.min(actor_q1, actor_q2)).mean()
                if actor_updated:
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    actor_updates += 1
                set_requires_grad(critic, True)
            soft_update(critic, target_critic, args.tau)

            if record:
                tensors = {"critic_loss": loss, "absolute_td_loss": absolute,
                           "differential_td_loss": differential, "actor_loss": actor_loss,
                           "q_mean": torch.min(q1, q2).mean(), "target_q_mean": target_q.mean()}
                row = {key: float(value.detach().cpu().item()) for key, value in tensors.items()}
                if args.representation in DIAGNOSTIC_REPRESENTATIONS:
                    for name, model in (("actor", actor), ("critic", critic)):
                        diagnostics = model.history_encoder.last_diagnostics or {}
                        row.update({name + "_" + key: float(value.cpu().item())
                                    for key, value in diagnostics.items()})
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError("Non-finite training metric at step %d" % step)
                row["step"] = step
                row.update(actor_updated=actor_updated, actor_updates=actor_updates, critic_updates=step)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if step in checkpoint_steps:
                checkpoint = dict(metadata, actor=actor.state_dict(), critic=critic.state_dict(),
                                  target_critic=target_critic.state_dict(), normalization=normalization,
                                  model_config=model_config, train_config=vars(args), step=step,
                                  actor_updates=actor_updates, critic_updates=step, target_updates=step)
                atomic_checkpoint(output / checkpoint_name(step, args.train_steps), checkpoint)

    write_json(output / "training_complete.json", dict(
        metadata, seed=args.seed, train_steps=args.train_steps,
        checkpoint_interval=args.checkpoint_interval, saved_steps=sorted(checkpoint_steps),
        num_transitions=len(dataset), device=str(device), dataset=str(args.dataset),
        model_config=model_config, actor_updates=actor_updates,
        critic_updates=args.train_steps, target_updates=args.train_steps))
    print("Saved model: %s" % (output / "model.pt"), flush=True)
    return str(output / "model.pt")
