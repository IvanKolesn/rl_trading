"""
Torch model for RayLib actions
"""

import numpy as np
import ray
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

        self.main_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_net = nn.Sequential(nn.Linear(64, num_outputs), nn.Tanh())

        self.value_net = nn.Linear(64, 1)
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        x = self.main_net(input_dict["obs"])
        self._value = self.value_net(x)
        return self.action_net(x), state

    def value_function(self):
        return self._value.squeeze(1)
