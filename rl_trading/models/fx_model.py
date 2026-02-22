"""
Torch model for RayLib actions
"""

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class FXModel(TorchModelV2, nn.Module):
    """
    Model for action prediction
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.main_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.mean_net = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
        self.log_std_net = nn.Sequential(nn.Linear(64, action_dim))

        self.value_net = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass
        """
        x = self.main_net(input_dict["obs"].float())
        self._value = self.value_net(x)

        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, min=-3.0, max=1.0)
        actions = torch.cat([mean, log_std], dim=-1)

        return actions, state

    def value_function(self):
        return self._value.squeeze(1)
