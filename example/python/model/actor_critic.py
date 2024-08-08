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
                            hidden_features=config["actor_hidden_dim"],
                            out_features=num_actions,
                            n_layers=config["actor_n_layers"],
                            act=nn.ELU(),
                            output_act=nn.Tanh(),
                            using_norm=False)

        self.critic = MLP(in_features=num_priv_obs,
                          hidden_features=config["critic_hidden_dim"],
                          out_features=1,
                          n_layers=config["critic_n_layers"],
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # Action distribution
        self.std_action = nn.Parameter(config["std"] * torch.ones(num_actions))
        self.distribution_action = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution_action.mean

    @property
    def action_std(self):
        return self.distribution_action.stddev

    @property
    def entropy(self):
        return self.distribution_action.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean_action = self.actor(observations)
        self.distribution_action = Normal(mean_action, torch.abs(self.std_action))
        

    def act(self, observations,**kwargs):
        if self.use_encoder_decoder:
            # encode
            latent = self.encoder(observations.reshape(-1, self.num_obs // self.num_single_obs ,self.num_single_obs)) # batch_size, seq_len, num_single_obs
            # actor
            self.update_distribution(latent)
            return self.distribution_action.sample(), self.decoder(latent) # policy, state estimation
        elif self.use_lstm:
            self.update_distribution(observations.reshape(-1, self.num_obs // self.num_single_obs ,self.num_single_obs))
            return self.distribution_action.sample()
        else:
            self.update_distribution(observations)
            return self.distribution_action.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution_action.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    ## for recurrent encoder/decoder
    def get_states(self, observations):
        latent=  self.encoder(observations.reshape(-1, self.num_obs // self.num_single_obs ,self.num_single_obs))
        #self.update_distribution(latent)
        return latent, self.decoder(latent)