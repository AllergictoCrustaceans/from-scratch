# citing my references for learning about PPO:
# https://huggingface.co/learn/deep-rl-course/unit8/visualize
# https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal

class SharedActorCritic(nn.Module):
    def __init__(self):
        super(SharedActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.act_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(1281, 1)
        )   
        
        # create variable for the matrix
        self.cov_var = torch.full(size=(self.act_dim, ), fill_value=0.5)
        # create covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
            
    # helper function for STEP 2c       
    def get_actions(self, obs):
        # query the actor network for a mean action
        mean = self.actor(obs)
        
        # create our multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        
        # sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # return the sampled action and the log prob of that action
        return action.detach().numpy(), log_prob.detach()
    
    # helper function for STEP 2d
    def evaluate(self, batch_obs, batch_acts):
        # query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()
        
        # caluclate the log probs of batch actions using most recent actor network
        # this block of code is similar to that of get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        return V, log_probs
    
    
    
class PPO:
    def __init__(self, env):
        self.env = env
        
        # STEP 1: init actor and critic networks
        self.policy = SharedActorCritic(env)
        
        # init optimizers
        self.actor_optim = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr)
        
        # init hyperparameters
        self._init_hyperparameters()
        
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
      
      
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 500000
        self.max_timesteps_per_episode = 1000
        self.n_updates_per_iteration = 5
        self.gamma = 0.95
        self.clip = 0.2
        self.lr = 1e-3     
        
      
    # STEP 2: begin iterative training    
    def train(self, total_timesteps):
        k = 0 # timesteps simulated so far
        while k < total_timesteps:
            # STEP 2a & 2b
            batch_obs, batch_acts, batch_log_probs, batch_reward_to_go, batch_lens = self.rollout()
            
            k += np.sum(batch_lens)
            
            # STEP 2c
            # calculate V_{phi, k}
            V, _ = self.policy.evaluate(batch_obs, batch_acts)
            # calculate advantage
            A_k = batch_reward_to_go - V.detach()

            # normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            # STEP 2d & 2e
            for _ in range(self.n_updates_per_iteration):
                # calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.policy.evaluate(batch_obs, batch_acts)
                
                # caluclate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                # caluclate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k 
                
                # caluclate actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_reward_to_go) 
                
                # caluclate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.critic_optim.step()
                
                # calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()          
        
        
    # helper for STEP 2a
    def rollout(self):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_reward_to_go = []
        batch_lens = []
        
        # number of timesteps run so far in this batch
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment t
                t += 1
                
                # collect observations
                batch_obs.append(obs)
                action, log_prob = self.policy.get_actions(obs)
                obs, rew, done, _ = self.env.step(action)
                
                # collect reward, action, log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                if done: break
            
            # collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        # 2b
        batch_reward_to_go = self.compute_reward_to_go(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_reward_to_go, batch_lens
     
    
    # helper function for STEP 2b
    def compute_reward_to_go(self, batch_rews):
        # the rewards-to-go per episode per batch to return
        # the shape will be (number of timesteps per episode)
        batch_reward_to_go = []
        
        # iterate through episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_reward_to_go.insert(0, discounted_reward)
            
        # conver the rewards-to-go into a tensor
        batch_reward_to_go = torch.tensor(batch_reward_to_go, dtype=torch.float)
        return batch_reward_to_go
    
    
 
    