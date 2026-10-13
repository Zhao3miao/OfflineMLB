"""CIO-query cross-attention over users inside each directed residual Link.

Each link concatenates ordinary user mean and CIO-query attention summaries.
Keys/values share the original user MLP; each query sees only its own
frame's forward/reverse CIO. No user identity, extra features or N-by-N attention.
"""
import math

import torch
from torch import nn

from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic
from agent.history_fusion_link_sac import HistoryFusionLinkEncoder
from agent.link_snapshot_sac import mlp

REPRESENTATION = "user_attention_residual_link_snapshot_sac"
MEAN_ATTENTION_POOLING = "mean_plus_cio_query_user_attention"


class CIOUserAttentionPool(nn.Module):
    def __init__(self, dimension, heads=2):
        super().__init__()
        if min(dimension, heads) < 1 or dimension % heads:
            raise ValueError("summary-dim must be positive and divisible by attention-heads")
        self.dimension, self.heads = dimension, heads
        self.head_dim = dimension // heads
        self.query = nn.Sequential(nn.Linear(2, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, tokens, valid_users, pair_cio):
        # tokens [B,H,N,6,D]; query [B,H,6,heads,head_dim].
        batch, history, users, links, dim = tokens.shape
        if dim != self.dimension or tuple(pair_cio.shape) != (batch, history, links, 2):
            raise ValueError("Invalid user token/query dimensions")
        if tuple(valid_users.shape) != (batch, history, users):
            raise ValueError("Invalid user mask dimensions")
        if users == 0:
            return tokens.sum(dim=2)
        query = self.query(pair_cio).reshape(batch, history, links, self.heads, self.head_dim)
        key = self.key(tokens).reshape(batch, history, users, links, self.heads, self.head_dim)
        value = self.value(tokens).reshape(batch, history, users, links, self.heads, self.head_dim)
        logits = torch.einsum("bhlkd,bhnlkd->bhlkn", query, key) / math.sqrt(self.head_dim)
        mask = valid_users[:, :, None, None, :]
        # Finite fill plus explicit masking handles real empty and padded frames.
        weights = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1) * mask.to(logits.dtype)
        summary = torch.einsum("bhlkn,bhnlkd->bhlkd", weights, value)
        summary = self.output(summary.reshape(batch, history, links, dim))
        return summary * valid_users.any(dim=2)[:, :, None, None].to(summary.dtype)


class UserAttentionLinkEncoder(HistoryFusionLinkEncoder):
    def __init__(self, token_dim=9, global_dim=7, history_length=5,
                 summary_dim=32, temporal_dim=64, fusion="residual", attention_heads=2,
                 link_pooling=MEAN_ATTENTION_POOLING):
        if fusion != "residual":
            raise ValueError("User attention uses residual history fusion")
        if link_pooling != MEAN_ATTENTION_POOLING:
            raise ValueError("Expected the full mean-plus-attention Link representation")
        super().__init__(token_dim, global_dim, history_length, summary_dim, temporal_dim, fusion)
        self.link_pooling = link_pooling
        self.user_attention = CIOUserAttentionPool(summary_dim, attention_heads)
        self.link_fusion = nn.Sequential(nn.Linear(2 * summary_dim, summary_dim), nn.ReLU())

    def pool_links(self, link_tokens, valid_users, global_obs, count):
        cio = global_obs[..., :6]
        pair_cio = torch.stack([cio, cio[..., self.reverse_links]], dim=-1)
        attended = self.user_attention(link_tokens, valid_users, pair_cio)
        mean = super().pool_links(link_tokens, valid_users, global_obs, count)
        links = self.link_fusion(torch.cat([mean, attended], dim=-1))
        return links * valid_users.any(dim=2)[:, :, None, None].to(links.dtype)


class UserAttentionLinkActor(HistoryAttentionSACActor):
    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128, fusion="residual", attention_heads=2,
                 link_pooling=MEAN_ATTENTION_POOLING):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed actions")
        self.history_encoder = UserAttentionLinkEncoder(
            token_dim, global_dim, history_length, summary_dim, temporal_dim, fusion, attention_heads,
            link_pooling)
        self.body = mlp(self.history_encoder.output_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


class UserAttentionLinkCritic(HistoryAttentionSACCritic):
    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128, fusion="residual", attention_heads=2,
                 link_pooling=MEAN_ATTENTION_POOLING):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed actions")
        self.history_encoder = UserAttentionLinkEncoder(
            token_dim, global_dim, history_length, summary_dim, temporal_dim, fusion, attention_heads,
            link_pooling)
        size = self.history_encoder.output_dim + action_dim
        self.q1 = nn.Sequential(mlp(size, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(mlp(size, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))
