"""IQL with SAC-aligned set/GRU actor and twin Q, plus an independent value net.

The bounded Gaussian actor fits advantage-weighted dataset-action likelihood.
Neither the TD target nor actor objective queries policy-action Q.
"""
import math

import torch
from torch import nn
from torch.distributions import Normal

from agent.attention_sac import AttentionSetEncoder, critic_loss, soft_update
from agent.history_attention_sac import HistorySetEncoder
from agent.offline_common import SACUpdate, make_networks, encode_state, optimize, require_finite_positive


def expectile_loss(residual, expectile):
    return (torch.where(residual > 0, expectile, 1 - expectile) * residual.square()).mean()


def advantage_weights(advantage, inverse_temperature, max_weight):
    return torch.exp((inverse_temperature * advantage.detach()).clamp(max=math.log(max_weight)))


def dataset_action_log_prob(actor, inputs, actions):
    mean, log_std = actor.distribution(*inputs)
    bounded = actions.clamp(-1 + 1e-6, 1 - 1e-6)
    raw = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
    return (Normal(mean, log_std.exp()).log_prob(raw)
            - torch.log(1 - bounded.square() + 1e-6)).sum(-1, keepdim=True)


class IQLValue(nn.Module):
    def __init__(self, token_dim, global_dim, action_dim, embedding_dim=128,
                 hidden_dim=256, num_heads=4, history_length=1):
        super().__init__()
        if history_length == 1:
            self.encoder = AttentionSetEncoder(token_dim, embedding_dim, embedding_dim, num_heads)
        else:
            self.history_encoder = HistorySetEncoder(token_dim, global_dim, history_length,
                                                      embedding_dim, num_heads)
        self.head = nn.Sequential(nn.Linear(embedding_dim + global_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, *inputs):
        return self.head(encode_state(self, inputs))


class IQLAgent(SACUpdate):
    def __init__(self, model_config, device="cpu", expectile=0.7,
                 inverse_temperature=3.0, max_weight=100.0, value_lr=3e-4, **kwargs):
        if not 0 < expectile < 1:
            raise ValueError("expectile must be in (0,1)")
        require_finite_positive(inverse_temperature=inverse_temperature,
                                max_weight=max_weight, value_lr=value_lr)
        actor, critic = make_networks(model_config)
        super().__init__(actor, critic, model_config.get("history_length", 1), device,
                         alpha=0.0, **kwargs)
        self.value = IQLValue(**model_config).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.expectile, self.inverse_temperature, self.max_weight = expectile, inverse_temperature, max_weight
        self.value_updates = 0

    def update(self, batch):
        current, following = self.inputs(batch), self.inputs(batch, True)
        with torch.no_grad():
            tq1, tq2 = self.target_critic(*current, batch["action"])
            data_q = torch.min(tq1, tq2)
        value = self.value(*current)
        advantage = data_q - value
        v_loss = expectile_loss(advantage, self.expectile)
        weights = advantage_weights(advantage, self.inverse_temperature, self.max_weight)
        optimize(v_loss, self.value_optimizer, self.value, self.grad_clip)
        self.value_updates += 1
        with torch.no_grad():
            target = self.reward_scale * batch["reward"] + self.gamma * (1 - batch["terminal"]) * self.value(*following)
        q1, q2 = self.critic(*current, batch["action"])
        q_loss = critic_loss(q1, q2, target)
        optimize(q_loss, self.critic_optimizer, self.critic, self.grad_clip)
        self.critic_updates += 1
        actor_updated = self.critic_updates % self.actor_update_interval == 0
        with torch.set_grad_enabled(actor_updated):
            logp = dataset_action_log_prob(self.actor, current, batch["action"])
            actor_loss = -(weights * logp).mean()
        if actor_updated:
            optimize(actor_loss, self.actor_optimizer, self.actor, self.grad_clip)
            self.actor_updates += 1
        soft_update(self.critic, self.target_critic, self.tau)
        self.target_updates += 1
        return {key: val.detach() for key, val in dict(
            critic_loss=q_loss, absolute_td_loss=q_loss, actor_loss=actor_loss,
            value_loss=v_loss, value_mean=value.mean(), q_mean=torch.min(q1, q2).mean(),
            target_q_mean=target.mean(), advantage_mean=advantage.mean(),
            advantage_weight_mean=weights.mean(), advantage_weight_max=weights.max(),
            actor_updated=q_loss.new_tensor(actor_updated),
            actor_updates=q_loss.new_tensor(self.actor_updates),
            critic_updates=q_loss.new_tensor(self.critic_updates)).items()}

    def update_counts(self):
        return dict(super().update_counts(), value_updates=self.value_updates)

    def state_dict(self):
        result = super().state_dict()
        result.update(value=self.value.state_dict(), value_optimizer=self.value_optimizer.state_dict())
        return result
