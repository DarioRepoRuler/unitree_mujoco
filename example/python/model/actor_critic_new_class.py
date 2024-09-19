import numpy as np
import torch
import torch.nn as nn
from model.networks import *
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self,
                 config,
                 num_single_obs,
                 num_obs,
                 num_priv_obs,
                 num_actions):
        super().__init__()

        self.num_single_obs = num_single_obs
        self.num_obs = num_obs
        self.num_priv_obs = num_priv_obs

        self.actor = MLP(in_features=num_obs,
                            hidden_features=config.actor.hidden_dim,
                            out_features=num_actions,
                            n_layers=config.actor.n_layers,
                            act=nn.ELU(),
                            output_act=nn.Tanh(),
                            using_norm=False)

        self.critic = MLP(in_features=num_priv_obs,
                          hidden_features=config.critic.hidden_dim,
                          out_features=1,
                          n_layers=config.critic.n_layers,
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # Action distribution
        self.std_action = nn.Parameter(config.actor.init_std * torch.ones(num_actions))
        self.distribution_action = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False



    def forward(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
