"""Shared SAC networks and update utilities for offline baselines.

H=1 uses existing SAC classes verbatim; H>1 uses History SAC's set+GRU
classes verbatim. Inputs come from the existing history dataset/collator.
"""
import math

import torch
from torch.distributions import Normal

from agent.attention_sac import (
    AttentionSACActor, AttentionSACCritic, LOG_STD_MIN, LOG_STD_MAX,
    critic_loss, make_target_critic, soft_update,
)
from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic


def make_networks(config):
    config = dict(config)
    history = config.pop("history_length", 1)
    if history == 1:
        return AttentionSACActor(**config), AttentionSACCritic(**config)
    return (HistoryAttentionSACActor(history_length=history, **config),
            HistoryAttentionSACCritic(history_length=history, **config))


def batch_inputs(batch, history_length, following=False):
    prefix = "next_" if following else ""
    values = tuple(batch[prefix + key] for key in
                   ("user_history", "user_mask", "global_history", "history_mask"))
    return tuple(value[:, -1] for value in values[:3]) if history_length == 1 else values


def encode_state(model, inputs):
    if hasattr(model, "history_encoder"):
        return model.history_encoder(*inputs)
    return torch.cat([model.encoder(inputs[0], inputs[1]), inputs[2]], dim=-1)


def sample_encoded(actor, encoded, num_actions):
    """Sample K actions without re-encoding K copies of each history."""
    hidden = actor.body(encoded)
    mean = actor.mean(hidden)[:, None, :].expand(-1, num_actions, -1)
    log_std = actor.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)[:, None, :].expand_as(mean)
    dist = Normal(mean, log_std.exp())
    raw = dist.rsample()
    action = raw.tanh()
    logp = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
    return action, logp


def encoded_q(critic, encoded, actions):
    if actions.ndim == 3:
        encoded = encoded[:, None, :].expand(-1, actions.shape[1], -1)
    features = torch.cat([encoded, actions], dim=-1)
    return critic.q1(features), critic.q2(features)


def optimize(loss, optimizer, module, grad_clip):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
    optimizer.step()


def require_finite_positive(**values):
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("Expected positive finite values: %s" % values)


class SACUpdate:
    """SAC update with existing networks and already-collated offline batches."""
    def __init__(self, actor, critic, history_length=1, device="cpu", actor_lr=3e-4,
                 critic_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 reward_scale=1.0, grad_clip=5.0, actor_update_interval=1):
        require_finite_positive(actor_lr=actor_lr, critic_lr=critic_lr,
                                reward_scale=reward_scale, grad_clip=grad_clip)
        if not 0 <= gamma <= 1 or not 0 < tau <= 1 or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid gamma, tau, or alpha")
        if actor_update_interval < 1 or int(actor_update_interval) != actor_update_interval:
            raise ValueError("actor_update_interval must be a positive integer")
        self.actor_update_interval = int(actor_update_interval)
        self.critic_updates = self.actor_updates = self.target_updates = 0
        self.history_length = history_length
        self.device = torch.device(device)
        self.actor, self.critic = actor.to(self.device), critic.to(self.device)
        self.target_critic = make_target_critic(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.reward_scale, self.grad_clip = reward_scale, grad_clip

    def inputs(self, batch, following=False):
        return batch_inputs(batch, self.history_length, following)

    def critic_objective(self, current, following, actions, target):
        q1, q2 = self.critic(*current, actions)
        td = critic_loss(q1, q2, target)
        return td, dict(absolute_td_loss=td, q_mean=torch.min(q1, q2).mean())

    def update(self, batch):
        current, following = self.inputs(batch), self.inputs(batch, True)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(*following)
            q1, q2 = self.target_critic(*following, next_action)
            target = self.reward_scale * batch["reward"] + self.gamma * (1 - batch["terminal"]) * (
                torch.min(q1, q2) - self.alpha * next_logp)
        loss, metrics = self.critic_objective(current, following, batch["action"], target)
        optimize(loss, self.critic_optimizer, self.critic, self.grad_clip)
        self.critic_updates += 1
        actor_updated = self.critic_updates % self.actor_update_interval == 0
        for p in self.critic.parameters():
            p.requires_grad_(False)
        try:
            with torch.set_grad_enabled(actor_updated):
                action, logp = self.actor.sample(*current)
                q1, q2 = self.critic(*current, action)
                actor_q = torch.min(q1, q2)
                actor_loss = (self.alpha * logp - actor_q).mean()
            if actor_updated:
                optimize(actor_loss, self.actor_optimizer, self.actor, self.grad_clip)
                self.actor_updates += 1
        finally:
            for p in self.critic.parameters():
                p.requires_grad_(True)
        soft_update(self.critic, self.target_critic, self.tau)
        self.target_updates += 1
        metrics.update(critic_loss=loss, actor_loss=actor_loss, target_q_mean=target.mean(),
                       actor_q_mean=actor_q.mean(), entropy_term=(self.alpha * logp).mean(),
                       actor_updated=loss.new_tensor(actor_updated),
                       actor_updates=loss.new_tensor(self.actor_updates),
                       critic_updates=loss.new_tensor(self.critic_updates))
        return {key: value.detach() for key, value in metrics.items()}

    def update_counts(self):
        return dict(actor_update_interval=self.actor_update_interval,
                    critic_updates=self.critic_updates, actor_updates=self.actor_updates,
                    target_update_interval=1, target_updates=self.target_updates)

    def state_dict(self):
        result = {key: getattr(self, key).state_dict() for key in
                  ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer")}
        result.update(self.update_counts())
        return result
