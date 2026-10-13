"""Continuous-action CQL(H) using the project's SAC or History SAC networks.

Fixed conservative weight, twin critics, importance-corrected uniform/current-
policy/next-policy proposals. No PARL or online replay buffer is required.
"""
import math

import torch

from agent.attention_sac import critic_loss
from agent.offline_common import (
    SACUpdate, make_networks, encode_state, encoded_q, sample_encoded,
    require_finite_positive,
)


def conservative_penalty(candidate_q, proposal_log_density, data_q, temperature=1.0):
    """Estimate T log integral exp(Q/T) da minus Q(s,a_data).

    Proposal densities enter as Q/T - log p. Sample count normalization changes
    only an additive constant for the fixed conservative weight used here.
    """
    require_finite_positive(temperature=temperature)
    corrected = candidate_q / temperature - proposal_log_density.detach()
    return (temperature * (torch.logsumexp(corrected, dim=1)
                           - math.log(candidate_q.shape[1])) - data_q).mean()


class CQLAgent(SACUpdate):
    def __init__(self, model_config, device="cpu", cql_weight=1.0,
                 cql_temperature=1.0, cql_num_actions=10, **kwargs):
        if not math.isfinite(cql_weight) or cql_weight < 0:
            raise ValueError("cql_weight must be finite and nonnegative")
        require_finite_positive(cql_temperature=cql_temperature)
        if cql_num_actions < 1 or int(cql_num_actions) != cql_num_actions:
            raise ValueError("cql_num_actions must be a positive integer")
        actor, critic = make_networks(model_config)
        super().__init__(actor, critic, model_config.get("history_length", 1), device, **kwargs)
        self.action_dim = model_config["action_dim"]
        self.cql_weight, self.cql_temperature = cql_weight, cql_temperature
        self.cql_num_actions = int(cql_num_actions)

    def critic_objective(self, current, following, actions, target):
        encoded = encode_state(self.critic, current)
        q1, q2 = encoded_q(self.critic, encoded, actions)
        td = critic_loss(q1, q2, target)
        penalty = td.new_zeros(())
        if self.cql_weight:
            batch_size = actions.shape[0]
            with torch.no_grad():
                random_actions = actions.new_empty(batch_size, self.cql_num_actions,
                                                   self.action_dim).uniform_(-1, 1)
                current_actions, current_logp = sample_encoded(
                    self.actor, encode_state(self.actor, current), self.cql_num_actions)
                next_actions, next_logp = sample_encoded(
                    self.actor, encode_state(self.actor, following), self.cql_num_actions)
                proposals = torch.cat([random_actions, current_actions, next_actions], dim=1)
                log_density = torch.cat([torch.full_like(current_logp, -self.action_dim * math.log(2)),
                                         current_logp, next_logp], dim=1)
            # Even a~pi(.|s') proposals are evaluated at CURRENT state s.
            candidates1, candidates2 = encoded_q(self.critic, encoded, proposals)
            penalty = (conservative_penalty(candidates1, log_density, q1, self.cql_temperature)
                       + conservative_penalty(candidates2, log_density, q2, self.cql_temperature))
        return td + self.cql_weight * penalty, dict(
            absolute_td_loss=td, cql_loss=penalty, q_mean=torch.min(q1, q2).mean())
