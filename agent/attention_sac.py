import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class AttentionSetEncoder(nn.Module):
    """Permutation-invariant encoder for a padded set of active users."""

    def __init__(self, token_dim, embedding_dim=128, hidden_dim=128, num_heads=4):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.token_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_obs, user_mask):
        batch_size = user_obs.shape[0]
        token_embeddings = self.token_net(user_obs)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, token_embeddings], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=user_obs.device)
        padding_mask = torch.cat([cls_mask, ~user_mask.bool()], dim=1)
        encoded = self.transformer(
            tokens.transpose(0, 1),
            src_key_padding_mask=padding_mask,
        )
        return self.output(encoded[0])


class AttentionSACActor(nn.Module):
    def __init__(self, token_dim, global_dim, action_dim, embedding_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = AttentionSetEncoder(token_dim, embedding_dim, embedding_dim, num_heads)
        self.body = nn.Sequential(
            nn.Linear(embedding_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, user_obs, user_mask, global_obs):
        encoded = self.encoder(user_obs, user_mask)
        hidden = self.body(torch.cat([encoded, global_obs], dim=-1))
        return self.mean(hidden), torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, user_obs, user_mask, global_obs, deterministic=False):
        mean, log_std = self.distribution(user_obs, user_mask, global_obs)
        if deterministic:
            return torch.tanh(mean), None
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class AttentionSACCritic(nn.Module):
    def __init__(self, token_dim, global_dim, action_dim, embedding_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = AttentionSetEncoder(token_dim, embedding_dim, embedding_dim, num_heads)
        input_dim = embedding_dim + global_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_obs, user_mask, global_obs, action):
        encoded = self.encoder(user_obs, user_mask)
        features = torch.cat([encoded, global_obs, action], dim=-1)
        return self.q1(features), self.q2(features)


def soft_update(source, target, tau):
    with torch.no_grad():
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)


def make_target_critic(critic):
    target = copy.deepcopy(critic)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def critic_loss(q1, q2, target):
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)
