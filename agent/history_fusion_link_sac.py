"""Current-frame residual or gated fusion of the original joint Link history.

Both paths reuse LinkSnapshotEncoder's frame features, including association
and CIO. History includes the current frame. Without user-level CIO conditioning,
H=1 is exactly the original Link architecture. Actor and critic own separate encoders.
"""
import torch
from torch import nn

from agent.history_attention_sac import HistoryAttentionSACActor, HistoryAttentionSACCritic
from agent.link_snapshot_sac import LinkSnapshotEncoder, mlp


CIO_CONDITIONED_REPRESENTATION = "cio_conditioned_residual_link_snapshot_sac"

FUSION_REPRESENTATIONS = {
    "residual_link_snapshot_sac": "residual",
    "gated_link_snapshot_sac": "gated",
    "action_conditioned_residual_link_snapshot_sac": "residual",
    "action_conditioned_gated_link_snapshot_sac": "gated",
    CIO_CONDITIONED_REPRESENTATION: "residual",
}


class HistoryFusionLinkEncoder(LinkSnapshotEncoder):
    def __init__(self, token_dim=9, global_dim=7, history_length=5,
                 summary_dim=32, temporal_dim=64, fusion="gated", user_cio_conditioned=False):
        if fusion not in ("residual", "gated"):
            raise ValueError("fusion must be residual or gated")
        super().__init__(token_dim, global_dim, history_length, summary_dim, temporal_dim,
                         user_cio_conditioned=user_cio_conditioned)
        self.fusion = fusion
        self.history_projection = None
        self.history_gate = None
        self.history_enabled = True  # Explicit evaluation ablation, not a learned parameter.
        self.record_diagnostics = False
        self.last_diagnostics = None
        if history_length > 1:
            self.history_projection = nn.Linear(temporal_dim, temporal_dim, bias=False)
            if fusion == "gated":
                # The extra gate must not shift shared actor/critic initialization.
                with torch.random.fork_rng(devices=[]):
                    self.history_gate = nn.Linear(2 * temporal_dim, temporal_dim)
                    nn.init.zeros_(self.history_gate.weight)
                    nn.init.constant_(self.history_gate.bias, -2.0)

    def forward(self, user_history, user_mask, global_history, history_mask):
        frames, population, globals_, valid = self.encode_frames(
            user_history, user_mask, global_history, history_mask)
        current = frames[:, -1]
        self.last_diagnostics = None
        if self.temporal is not None and self.history_enabled:
            hidden = self.encode_history(frames, valid)
            gate = (torch.sigmoid(self.history_gate(torch.cat([current, hidden], dim=-1)))
                    if self.history_gate is not None else torch.ones_like(current))
            contribution = gate * self.history_projection(hidden)
            # Detached diagnostics retain neither the graph nor the full history.
            with torch.no_grad():
                observed = valid[:, -1]
                if self.record_diagnostics and observed.any():
                    active_gate = gate[observed]
                    self.last_diagnostics = {
                        "gate_mean": active_gate.mean(),
                        "gate_std": active_gate.std(unbiased=False),
                        "current_abs_mean": current[observed].abs().mean(),
                        "history_abs_mean": contribution[observed].abs().mean(),
                    }
                    for link in range(6):
                        self.last_diagnostics["gate_link_%d_mean" % link] = active_gate[:, link].mean()
            current = current + contribution
        current = torch.where(valid[:, -1, None, None], current, torch.zeros_like(current))
        return torch.cat([current.reshape(frames.shape[0], -1), population[:, -1], globals_[:, -1]], dim=-1)


class HistoryFusionLinkSACActor(HistoryAttentionSACActor):
    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128, fusion="gated",
                 critic_action_conditioned=False, user_cio_conditioned=False):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed CIO actions")
        self.history_encoder = HistoryFusionLinkEncoder(
            token_dim, global_dim, history_length, summary_dim, temporal_dim, fusion,
            user_cio_conditioned=user_cio_conditioned)
        self.body = mlp(self.history_encoder.output_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


class HistoryFusionLinkSACCritic(HistoryAttentionSACCritic):
    def __init__(self, token_dim=9, global_dim=7, action_dim=6, history_length=5,
                 summary_dim=32, temporal_dim=64, hidden_dim=128, fusion="gated",
                 critic_action_conditioned=False, user_cio_conditioned=False):
        nn.Module.__init__(self)
        if action_dim != 6:
            raise ValueError("Expected six directed CIO actions")
        self.history_encoder = HistoryFusionLinkEncoder(
            token_dim, global_dim, history_length, summary_dim, temporal_dim, fusion,
            user_cio_conditioned=user_cio_conditioned)
        self.critic_action_conditioned = bool(critic_action_conditioned)
        self.action_link_projection = (nn.Linear(1, temporal_dim)
                                       if self.critic_action_conditioned else None)
        input_dim = self.history_encoder.output_dim + action_dim
        if self.critic_action_conditioned:
            input_dim += 6 * temporal_dim
        self.q1 = nn.Sequential(mlp(input_dim, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(mlp(input_dim, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))

    def _features(self, state, action, encoded=None):
        if encoded is None:
            encoded = self.history_encoder(*state)
        if self.critic_action_conditioned:
            action_links = self.action_link_projection(action.unsqueeze(-1)).reshape(action.shape[0], -1)
            return torch.cat([encoded, action_links, action], dim=-1)
        return torch.cat([encoded, action], dim=-1)

    def forward(self, user_history, user_mask, global_history, history_mask, action):
        features = self._features((user_history, user_mask, global_history, history_mask), action)
        return self.q1(features), self.q2(features)
