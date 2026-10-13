"""Lightweight directed-BS-pair population summaries with shared temporal encoding."""
import torch
from torch import nn

from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim), nn.ReLU())


class LinkSnapshotEncoder(nn.Module):
    """Encode unordered snapshots; temporal alignment is by fixed BS pair, not UE.

    Input token columns: [RSRP(3), measurement mask(3), association(3)].
    Global columns: six normalized CIOs in row-major directed-pair order, count.
    Only valid UE rows participate in pooling; unmeasured RSRP is zeroed while
    its measurement mask remains explicit. Real empty snapshots still advance
    temporal state; left-padding history frames do not.
    """

    def __init__(self, token_dim=9, global_dim=7, history_length=5,
                 summary_dim=32, temporal_dim=64, user_cio_conditioned=False):
        super().__init__()
        if token_dim != 9 or global_dim != 7:
            raise ValueError("Link snapshot encoder requires three BSs, 9 UE and 7 global features")
        if min(history_length, summary_dim, temporal_dim) < 1:
            raise ValueError("History and embedding dimensions must be positive")
        self.history_length = history_length
        self.user_cio_conditioned = bool(user_cio_conditioned)
        self.temporal_dim = temporal_dim
        self.output_dim = 6 * temporal_dim + summary_dim + global_dim
        pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
        self.register_buffer("link_features", torch.tensor([
            [i, j, 3 + i, 3 + j, 6 + i, 6 + j] for i, j in pairs
        ], dtype=torch.long))
        self.register_buffer("reverse_links", torch.tensor([
            pairs.index((j, i)) for i, j in pairs
        ], dtype=torch.long))
        self.link_net = mlp(8 if self.user_cio_conditioned else 6, summary_dim, summary_dim)
        self.population_net = mlp(9, summary_dim, summary_dim)
        self.frame_net = nn.Sequential(nn.Linear(2 * summary_dim + 3, temporal_dim), nn.ReLU())
        self.temporal = nn.GRUCell(temporal_dim, temporal_dim) if history_length > 1 else None

    def encode_frames(self, user_history, user_mask, global_history, history_mask):
        if user_history.ndim != 4 or user_history.shape[-1] != 9:
            raise ValueError("Expected user_history [B,H,N,9]")
        batch, history, users, _ = user_history.shape
        if (history != self.history_length or tuple(user_mask.shape) != (batch, history, users)
                or tuple(global_history.shape) != (batch, history, 7)
                or tuple(history_mask.shape) != (batch, history)):
            raise ValueError("Inconsistent history, user mask or global dimensions")
        valid_frames = history_mask.bool()
        valid_users = user_mask.bool() & valid_frames.unsqueeze(-1)
        safe = torch.where(valid_users.unsqueeze(-1), user_history, torch.zeros_like(user_history))
        safe_rsrp = torch.where(safe[..., 3:6] > 0, safe[..., :3], torch.zeros_like(safe[..., :3]))
        safe = torch.cat([safe_rsrp, safe[..., 3:]], dim=-1)
        global_obs = torch.where(valid_frames.unsqueeze(-1), global_history, torch.zeros_like(global_history))
        count = valid_users.sum(dim=2, keepdim=True).to(safe.dtype).clamp(min=1)

        link_inputs = safe[..., self.link_features]
        cio = global_obs[..., :6]
        if self.user_cio_conditioned:
            # Each history frame uses its own observed controls. Reverse links
            # see the same two CIOs in the opposite order, before user pooling.
            pair_cio = torch.stack([cio, cio[..., self.reverse_links]], dim=-1)
            pair_cio = pair_cio.unsqueeze(2).expand(-1, -1, users, -1, -1)
            link_inputs = torch.cat([link_inputs, pair_cio], dim=-1)
        link_tokens = self.link_net(link_inputs)  # [B,H,N,6,D]
        links = self.pool_links(link_tokens, valid_users, global_obs, count)
        population_tokens = self.population_net(safe)
        population = (population_tokens * valid_users.unsqueeze(-1)).sum(dim=2) / count
        frames = self.frame_net(torch.cat([
            links,
            population.unsqueeze(-2).expand(-1, -1, 6, -1),
            cio.unsqueeze(-1),
            cio[..., self.reverse_links].unsqueeze(-1),
            global_obs[..., -1:].unsqueeze(-2).expand(-1, -1, 6, -1),
        ], dim=-1))
        return frames, population, global_obs, valid_frames

    def pool_links(self, link_tokens, valid_users, global_obs, count):
        """Summarize users before frame/history processing; default is unchanged."""
        return (link_tokens * valid_users[..., None, None]).sum(dim=2) / count.unsqueeze(-1)

    def encode_history(self, frames, valid_frames):
        batch, history = frames.shape[:2]
        if self.temporal is None:
            hidden = frames[:, 0] * valid_frames[:, 0, None, None]
        else:
            hidden = frames.new_zeros(batch, 6, self.temporal_dim)
            for t in range(history):
                candidate = self.temporal(
                    frames[:, t].reshape(batch * 6, self.temporal_dim),
                    hidden.reshape(batch * 6, self.temporal_dim),
                ).reshape(batch, 6, self.temporal_dim)
                hidden = torch.where(valid_frames[:, t, None, None], candidate, hidden)
        return hidden

    def forward(self, user_history, user_mask, global_history, history_mask):
        frames, population, global_obs, valid_frames = self.encode_frames(
            user_history, user_mask, global_history, history_mask)
        hidden = self.encode_history(frames, valid_frames)
        return torch.cat([hidden.reshape(frames.shape[0], -1), population[:, -1], global_obs[:, -1]], dim=-1)


class LinkSnapshotSACActor(HistoryAttentionSACActor):
    """Use the existing squashed-Gaussian SAC distribution with the new encoder."""

    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed CIO actions")
        self.history_encoder = LinkSnapshotEncoder(token_dim, global_dim, history_length,
                                                  summary_dim, temporal_dim)
        self.body = mlp(self.history_encoder.output_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


class LinkSnapshotSACCritic(HistoryAttentionSACCritic):
    """Separate from actor; twin Q heads jointly read all six pair histories."""

    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed CIO actions")
        self.history_encoder = LinkSnapshotEncoder(token_dim, global_dim, history_length,
                                                  summary_dim, temporal_dim)
        input_dim = self.history_encoder.output_dim + action_dim
        self.q1 = nn.Sequential(mlp(input_dim, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(mlp(input_dim, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))
