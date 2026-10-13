import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from agent.attention_sac import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    AttentionSetEncoder,
)


class HistorySetEncoder(nn.Module):
    """Encode a sequence of unordered UE sets without assuming UE identity."""

    def __init__(
        self,
        token_dim,
        global_dim,
        history_length=5,
        embedding_dim=128,
        num_heads=4,
    ):
        super().__init__()
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.history_length = int(history_length)
        self.frame_dim = int(embedding_dim + global_dim)
        self.set_encoder = AttentionSetEncoder(
            token_dim,
            embedding_dim,
            embedding_dim,
            num_heads,
        )
        self.temporal = (
            None
            if self.history_length == 1
            else nn.GRU(self.frame_dim, self.frame_dim, batch_first=True)
        )

    def forward(self, user_history, user_mask, global_history, history_mask):
        if user_history.ndim != 4:
            raise ValueError("user_history must have shape [B, H, N, D]")
        batch_size, history_length, num_users, token_dim = user_history.shape
        if history_length != self.history_length:
            raise ValueError(
                "Expected history length %d, got %d"
                % (self.history_length, history_length)
            )
        encoded_sets = self.set_encoder(
            user_history.reshape(batch_size * history_length, num_users, token_dim),
            user_mask.reshape(batch_size * history_length, num_users),
        ).reshape(batch_size, history_length, -1)
        frames = torch.cat([encoded_sets, global_history], dim=-1)
        if self.temporal is None:
            return frames[:, 0]

        hidden = torch.zeros(
            1,
            batch_size,
            self.frame_dim,
            dtype=frames.dtype,
            device=frames.device,
        )
        for index in range(history_length):
            _, candidate = self.temporal(frames[:, index : index + 1], hidden)
            valid = history_mask[:, index].reshape(1, batch_size, 1).bool()
            hidden = torch.where(valid, candidate, hidden)
        return hidden[0]


class HistoryAttentionSACActor(nn.Module):
    def __init__(
        self,
        token_dim,
        global_dim,
        action_dim,
        history_length=5,
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
    ):
        super().__init__()
        self.history_encoder = HistorySetEncoder(
            token_dim,
            global_dim,
            history_length,
            embedding_dim,
            num_heads,
        )
        frame_dim = embedding_dim + global_dim
        self.body = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, user_history, user_mask, global_history, history_mask):
        encoded = self.history_encoder(
            user_history, user_mask, global_history, history_mask
        )
        hidden = self.body(encoded)
        return self.mean(hidden), torch.clamp(
            self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX
        )

    def sample(
        self,
        user_history,
        user_mask,
        global_history,
        history_mask,
        deterministic=False,
    ):
        mean, log_std = self.distribution(
            user_history, user_mask, global_history, history_mask
        )
        if deterministic:
            return torch.tanh(mean), None
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class HistoryAttentionSACCritic(nn.Module):
    def __init__(
        self,
        token_dim,
        global_dim,
        action_dim,
        history_length=5,
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
    ):
        super().__init__()
        self.history_encoder = HistorySetEncoder(
            token_dim,
            global_dim,
            history_length,
            embedding_dim,
            num_heads,
        )
        input_dim = embedding_dim + global_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        user_history,
        user_mask,
        global_history,
        history_mask,
        action,
    ):
        encoded = self.history_encoder(
            user_history, user_mask, global_history, history_mask
        )
        features = torch.cat([encoded, action], dim=-1)
        return self.q1(features), self.q2(features)


class MatchedDifferentialHistorySACActor(HistoryAttentionSACActor):
    """Actor used by the matched-exogenous differential SAC model."""


class MatchedDifferentialHistorySACCritic(HistoryAttentionSACCritic):
    """Critic trained with absolute and matched-exogenous differential TD loss."""


def matched_differential_loss(q1, q2, target, differential_weight=1.0):
    if q1.shape[0] % 2:
        raise ValueError("Matched differential batches must contain an even sample count")
    absolute = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    target_difference = target[0::2] - target[1::2]
    q1_difference = q1[0::2] - q1[1::2]
    q2_difference = q2[0::2] - q2[1::2]
    differential = F.mse_loss(q1_difference, target_difference) + F.mse_loss(
        q2_difference, target_difference
    )
    total = absolute + float(differential_weight) * differential
    return total, absolute, differential
