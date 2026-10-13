"""History SAC alternative: unchanged UE-set inputs, temporal Transformer.

Each frame uses SAC's AttentionSetEncoder. Learned history position embeddings
and temporal attention replace the GRU. No UE identity alignment is assumed.
"""
import torch
from torch import nn

from agent.attention_sac import AttentionSetEncoder
from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic
from agent.offline_common import SACUpdate


class ObservationTransformerEncoder(nn.Module):
    def __init__(self, token_dim, global_dim, history_length=5, embedding_dim=128,
                 num_heads=4, temporal_dim=128, temporal_heads=4,
                 temporal_layers=1, temporal_feedforward_dim=256):
        super().__init__()
        if not 1 <= history_length <= 64 or temporal_layers < 1:
            raise ValueError("Invalid history length or temporal layer count")
        if temporal_heads < 1 or temporal_dim < 1 or temporal_dim % temporal_heads:
            raise ValueError("temporal_dim must be divisible by temporal_heads")
        self.history_length = history_length
        self.set_encoder = AttentionSetEncoder(token_dim, embedding_dim, embedding_dim, num_heads)
        self.frame_projection = nn.Linear(embedding_dim + global_dim, temporal_dim)
        self.position_embedding = nn.Parameter(torch.empty(history_length, temporal_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(temporal_dim, temporal_heads,
                                           temporal_feedforward_dim, dropout=0.0)
        self.temporal_encoder = nn.TransformerEncoder(layer, temporal_layers)
        self.output_projection = nn.Linear(temporal_dim, embedding_dim + global_dim)
        self.output_dim = embedding_dim + global_dim

    def forward(self, user_history, user_mask, global_history, history_mask):
        if user_history.ndim != 4:
            raise ValueError("Expected user_history [B,H,N,D]")
        batch, history, users, features = user_history.shape
        if (history != self.history_length or user_mask.shape != (batch, history, users)
                or history_mask.shape != (batch, history)
                or global_history.shape[:2] != (batch, history)):
            raise ValueError("Inconsistent history dimensions")
        valid = history_mask.bool()
        if not bool(valid[:, -1].all()):
            raise ValueError("Histories must end at a valid current frame")
        valid_users = user_mask.bool() & valid.unsqueeze(-1)
        safe_users = torch.where(valid_users.unsqueeze(-1), user_history, torch.zeros_like(user_history))
        safe_globals = torch.where(valid.unsqueeze(-1), global_history, torch.zeros_like(global_history))
        encoded = self.set_encoder(safe_users.reshape(batch * history, users, features),
                                   valid_users.reshape(batch * history, users)).reshape(batch, history, -1)
        frames = self.frame_projection(torch.cat([encoded, safe_globals], dim=-1))
        frames = frames + self.position_embedding.unsqueeze(0)
        # Sequence-first supports older PyTorch. All tokens are in the current
        # state's past, so temporal attention need not use a causal mask.
        encoded = self.temporal_encoder(frames.transpose(0, 1), src_key_padding_mask=~valid)
        return self.output_projection(encoded[-1])


class TransformerSACActor(HistoryAttentionSACActor):
    def __init__(self, token_dim, global_dim, action_dim, history_length=5,
                 embedding_dim=128, hidden_dim=256, num_heads=4,
                 temporal_dim=128, temporal_heads=4, temporal_layers=1,
                 temporal_feedforward_dim=256):
        nn.Module.__init__(self)
        self.history_encoder = ObservationTransformerEncoder(
            token_dim, global_dim, history_length, embedding_dim, num_heads,
            temporal_dim, temporal_heads, temporal_layers, temporal_feedforward_dim)
        self.body = nn.Sequential(nn.Linear(embedding_dim + global_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


class TransformerSACCritic(HistoryAttentionSACCritic):
    def __init__(self, token_dim, global_dim, action_dim, history_length=5,
                 embedding_dim=128, hidden_dim=256, num_heads=4,
                 temporal_dim=128, temporal_heads=4, temporal_layers=1,
                 temporal_feedforward_dim=256):
        nn.Module.__init__(self)
        self.history_encoder = ObservationTransformerEncoder(
            token_dim, global_dim, history_length, embedding_dim, num_heads,
            temporal_dim, temporal_heads, temporal_layers, temporal_feedforward_dim)
        width = embedding_dim + global_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


class TransformerSACAgent(SACUpdate):
    def __init__(self, model_config, device="cpu", **kwargs):
        super().__init__(TransformerSACActor(**model_config), TransformerSACCritic(**model_config),
                         model_config.get("history_length", 5), device, **kwargs)

    def inputs(self, batch, following=False):
        # Always consume history-shaped tensors, including H=1.
        prefix = "next_" if following else ""
        return tuple(batch[prefix + k] for k in
                     ("user_history", "user_mask", "global_history", "history_mask"))
