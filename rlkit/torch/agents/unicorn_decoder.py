"""
Faithful port of UNICORN's FOCALDecoder (UNICORN/models/decoder.py::FOCALDecoder).

The official repo uses `mbrl.models.GaussianMLP` with:
    deterministic=False, learn_logvar_bounds=True,
    num_layers=2, hid_size=aggregator_hidden_size(=128), ensemble_size=1
for both the dynamic and reward heads, and computes NLL on raw
(target_state - state) and raw target_reward. We replicate that exactly here
(ensemble_size is fixed to 1 — the official default for UNICORN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLPHead(nn.Module):
    def __init__(self, in_size, out_size, num_layers=2, hid_size=128):
        super().__init__()
        layers = []
        d = in_size
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hid_size))
            layers.append(nn.SiLU())
            d = hid_size
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hid_size, out_size)
        self.logvar_head = nn.Linear(hid_size, out_size)
        self.max_logvar = nn.Parameter(torch.ones(1, out_size) * 0.5)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_size) * 10.0)

    def forward(self, x):
        h = self.trunk(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, x, target):
        mean, logvar = self.forward(x)
        inv_var = torch.exp(-logvar)
        nll = ((mean - target) ** 2 * inv_var + logvar).mean()
        nll = nll + 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
        return nll


class FOCALGaussianDecoder(nn.Module):
    def __init__(self, obs_size, action_size, task_embedding_size,
                 num_layers=2, hidden_size=128):
        super().__init__()
        in_size = task_embedding_size + obs_size + action_size
        self.dynamic_decoder = GaussianMLPHead(in_size, obs_size, num_layers, hidden_size)
        self.reward_decoder = GaussianMLPHead(in_size, 1, num_layers, hidden_size)

    def loss(self, task_embedding, state, action, target_state, target_reward):
        x = torch.cat([task_embedding, state, action], dim=-1)
        state_loss = self.dynamic_decoder.loss(x, target_state - state)
        reward_loss = self.reward_decoder.loss(x, target_reward)
        return state_loss + reward_loss
