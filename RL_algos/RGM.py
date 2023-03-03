from ast import arg
import copy
from re import T
from tkinter import N
import torch
import torch.nn as nn
import numpy as np
import wandb
import gym
import os
import d4rl
import os, sys

import Sample_Dataset.Sample_from_dataset as Sample_from_dataset
from Sample_Dataset.Sample_from_dataset import DICEReplayBuffer
from Network.Actor_Critic_net import Actor, V_critic, V_with_r, Reward, Actor_deterministic
from torch.optim.lr_scheduler import CosineAnnealingLR
from RL_algos.discri import Discriminator, Discriminator_SA
from tqdm import tqdm
import datetime

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.dataset import ObsUtils

EPSILON = 1e-20
ADV_MIN = -100
ADV_MAX = 10

MUJOCO = ['hopper', 'walker2d', 'halfcheetah', 'ant']
ROBOMIMIC = ['lift', 'can', 'square', 'tool_hang', 'transport']
ADROID = ['door', 'hammer', 'pen', 'kitchen']
version = 'v2'

# Used for robomimic
obs_dict = {
    "low_dim": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object"
    ],
    "rgb": [                   
        "robot0_eye_in_hand_image",
        "agentview_image"
    ],
    "depth": [],
    "scan": []
}

obs_keys_low_dim=(                      # observations we want to appear in batches
    "robot0_eef_pos", 
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object"
)


def get_env_robomimic(dataset_path, render=False, obs_keys=None):
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_dict)
    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=obs_keys,
        verbose=True
    )
    env_names = [env_meta["env_name"]]
    for env_name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name, 
            render=render, 
            render_offscreen=False,
            use_image_obs=shape_meta["use_images"], 
        )
    return env


def transfer_state(s, obs_keys=None):
    obs_ = []
    for key in obs_keys:
        obs_.append(np.expand_dims(s[key], axis=1))
    return np.squeeze(np.concatenate(obs_, axis=0), axis=1)


class rgm:
    def __init__(self,
                 env_name: str,
                 dataset: str,
                 args: dict,
                 num_hidden: int = 256,
                 tau: float = 0.005,
                 seed: int = 0,
                 evalute_episodes: int = 10):
        super(rgm, self).__init__()

        # hyper-parameters
        self.num_hidden = args['num_hidden']
        self.gamma = args['gamma']
        self.tau = tau
        self.batch_size = args['batch_size']
        self.lr_actor = args['actor_lr']
        self.lr = args['lr']
        self.num_expert_traj = args['num_expert_traj']
        self.num_offline_traj = args['num_offline_traj']
        self.actor_deterministic = args['actor_deterministic']
        self.state = args['state']
        self.absorbing_state = args['absorbing_state']
        self.standardize_reward = args['standardize_reward']
        self.standardize_obs = args['standardize_obs']
        self.reward_type = args['reward_type']
        self.reward_scale = args['reward_scale']
        self.res_scale = args['res_scale']
        self.f = args['f']
        self.v_l2_reg = args['v_l2_reg']
        self.r_l2_reg = args['r_l2_reg']
        self.alpha = args['alpha']
        self.use_policy_entropy_constraint = args['use_policy_entropy_constraint']
        self.evaluate_freq = args['log_iterations']
        self.evaluate_episodes = args['episodes']
        self.device = args['device']
        self.total_it = 0
        self.disc_type = args['disc_type']
        self.disc_iteration = args['disc_iterations']
        self.render = args['render']

        # prepare the environment
        if env_name in ADROID:
            version = 'v0'
        else:
            version = 'v2'
        
        if env_name not in ROBOMIMIC:
            self.env_name = f"{env_name}-{dataset}-{version}"
            self.env = gym.make(self.env_name)
            if env_name == 'antmaze':
                self.expert_env = self.env
            elif env_name == 'kitchen':
                self.expert_env = gym.make(f"kitchen-complete-v0")
            else:
                self.expert_env = gym.make(f"{env_name}-expert-{version}")
        else:
            self.env_name = env_name
            dataset_path = f'/home/airtrans01/LJX/robomimic/datasets/{env_name}/ph/low_dim.hdf5'  # expert path
            self.obs_key=obs_keys_low_dim
            self.env = get_env_robomimic(dataset_path=dataset_path, render=self.render, obs_keys=self.obs_key)
            self.expert_env = self.env

        if env_name in ROBOMIMIC:
            if env_name == 'lift':
                num_state = 19
            if env_name == 'can':
                num_state = 23
            num_action = 7
        else:
            num_state = self.env.observation_space.shape[0]
            num_action = self.env.action_space.shape[0]

        # set seed
        self.seed = seed
        if env_name not in ROBOMIMIC:
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # prepare replay_buffer for dice and discriminator training
        self.replay_buffer = DICEReplayBuffer(env=self.env,
                                              env_name=self.env_name,
                                              expert_env=self.expert_env,
                                              batch_size=self.batch_size,
                                              state=self.state,
                                              num_offline_traj=self.num_offline_traj,
                                              num_expert_traj=self.num_expert_traj,
                                              standardize_observation=self.standardize_obs,
                                              absorbing_state=self.absorbing_state,
                                              standardize_reward=self.standardize_reward,
                                              reward_type=self.reward_type,
                                              robomimic_data_path=args['robomimic_path'],
                                              device=self.device)

        self.target_entropy = self.replay_buffer.prepare_dicedataset()

        # prepare the value, policy and discriminator network
        state_dim = num_state + 1 if self.absorbing_state else num_state
        action_dim = 0 if self.state else num_action
        disc_cutoff = state_dim

        self.discriminator = Discriminator_SA(disc_cutoff, action_dim, hidden_dim=self.num_hidden, env_name=env_name, device=self.device)

        if self.actor_deterministic:
            self.actor_net = Actor_deterministic(state_dim, num_action, self.num_hidden, self.device, env_name=env_name).float().to(self.device)
        else:
            self.actor_net = Actor(state_dim, num_action, self.num_hidden, self.device).float().to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optim, args['total_iterations'])

        self.v_net = V_critic(state_dim, self.num_hidden, self.device, env_name=env_name).float().to(self.device)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=self.lr, weight_decay=self.v_l2_reg)

        self.r_net = Reward(state_dim, num_action, self.num_hidden, self.res_scale, self.device, env_name=env_name).float().to(self.device)
        self.r_optim = torch.optim.Adam(self.r_net.parameters(), lr=self.lr * args['lr_ratio'],
                                        weight_decay=self.r_l2_reg)
        self.r_lr_schedule = CosineAnnealingLR(self.r_optim, args['total_iterations'])

        if self.use_policy_entropy_constraint and not self.actor_deterministic:
            self.log_ent_coeff = torch.zeros(1, requires_grad=True, device=self.device)
            self.ent_optim = torch.optim.Adam([self.log_ent_coeff], self.lr)

        # prepare f-divergence functions
        if self.f == 'chi':
            self.f_fn = lambda x: 0.5 * (x - 1) ** 2
            self.f_star_prime = lambda x: torch.relu(x + 1)
            self.f_star = lambda x: 0.5 * x ** 2 + x
        elif self.f == 'kl':
            self.f_fn = lambda x: x * torch.log(x + 1e-10)
            self.f_star_prime = lambda x: torch.exp(x - 1)
        elif self.f == 'soft-chi':
            self.f_fn = lambda x: torch.where(x < 1, x * torch.log(x + 1e-10) - x + 1, 0.5 * (x - 1) ** 2)
            self.f_star_prime = lambda x: torch.where(x < 0, torch.exp(torch.clamp(x, min=ADV_MIN)) + 1, x)
            self.f_star = lambda x: torch.where(x < 0, torch.exp(torch.clamp(x, min=ADV_MIN)), x + 1)  # TODO clip x ?
        else:
            raise NotImplementedError()

        # v and policy file location
        self.current_time = datetime.datetime.now()
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}"
        os.makedirs(logdir_name)
    

    def train_bc_only(self, total_time_step=1e+5):
        for total_it in tqdm(range(1, int(total_time_step) + 1)):
            loss_result = {}
            try:
                _, dice_batch = self.replay_buffer.sample_robomimic()
            except:
                self.replay_buffer.init_dataloader()
                _, dice_batch = self.replay_buffer.sample_robomimic()
                
            state = dice_batch['obs']
            action = dice_batch['actions'].squeeze(1).to(self.device)

            if self.actor_deterministic:
                action_pi = self.actor_net(state)
                bc_losses = torch.sum((action_pi - action)**2, dim=1, keepdim=True)
                policy_loss = torch.mean(bc_losses)
            
                self.actor_optim.zero_grad()
                policy_loss.backward()
                self.actor_optim.step()
                self.policy_lr_schedule.step()

                loss_result.update({'bc_loss': policy_loss.item()})
            else:
                raise NotImplementedError

            if total_it % self.evaluate_freq == 0:
                evaluate_reward = self.rollout_evaluate()
                loss_result.update({'evaluate_rewards': evaluate_reward})

            wandb.log(loss_result, step=total_it)


    def learn(self, total_time_step=1e+5):
        print("-------------------------------------------------------")
        print("discriminator training")
        self.pretrain_discriminator()

        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            init_state, state, action, reward_data, next_state, done, expert = self.replay_buffer._sample_minibatch(
                self.batch_size, self.reward_scale)
            init_state = init_state.to(self.device)
            state = state.to(self.device)
            action = action.to(self.device)
            pre_reward = reward_data.reshape(-1, 1).to(self.device)
            pre_reward[expert == 1] = 0
            next_state = next_state.to(self.device)
            done = done.reshape(-1, 1).to(self.device)

            # expert_index = (expert == 1).nonzero(as_tuple=False)
            # offline_index = (expert == 0).nonzero(as_tuple=False)
            #
            # pre_reward[expert_index] = 1
            # pre_reward[offline_index] = -1
            # get v loss, policy loss, r_loss and entropy loss
            # upper level optimization
            loss_result, reward_w_res = self.optim_reward(state, action, pre_reward, next_state, done, result={})

            # lower level optimization
            loss_result = self.get_V_policy_loss(init_state, state, action, pre_reward, next_state, done, loss_result)

            # lower level optimization step
            self.v_optim.zero_grad()
            loss_result['v_loss'].backward()
            self.v_optim.step()

            self.actor_optim.zero_grad()
            loss_result['policy_loss'].backward()
            self.actor_optim.step()
            self.policy_lr_schedule.step()

            if self.use_policy_entropy_constraint and not self.actor_deterministic:
                self.ent_optim.zero_grad()
                loss_result['ent_coeff_loss'].backward()
                self.ent_optim.step()

            # compute the important-weights for expert vs. offline data
            if 'w_e' in loss_result:
                if self.env_name in ROBOMIMIC:
                    expert = torch.zeros_like(loss_result['w_e'].cpu()).squeeze(1)
                expert_index = (expert == 1).nonzero(as_tuple=False)
                offline_index = (expert == 0).nonzero(as_tuple=False)
                w_e = loss_result['w_e']
                w_e_expert = w_e[expert_index].mean()
                w_e_offline = w_e[offline_index].mean()
                w_e_ratio = w_e_expert / w_e_offline
                w_e_overall = w_e.mean()
                pre_reward_expert = pre_reward[expert_index].mean()
                pre_reward_offline = pre_reward[offline_index].mean()
                reward_w_res_expert = reward_w_res[expert_index].mean()
                reward_w_res_offline = reward_w_res[offline_index].mean()

                loss_result.update({'w_e': w_e_overall,
                                    'w_e_expert': w_e_expert,
                                    'w_e_offline': w_e_offline,
                                    'w_e_ratio': w_e_ratio,
                                    'pre_r_expert': pre_reward_expert,
                                    'pre_r_offline': pre_reward_offline,
                                    'r_w_r_expert': reward_w_res_expert,
                                    'r_w_r_offline': reward_w_res_offline})

            if total_it % self.evaluate_freq == 0:
                evaluate_reward = self.rollout_evaluate()
                loss_result.update({'evaluate_rewards': evaluate_reward})

                print('reward:', evaluate_reward,
                        'w_ratio:', w_e_ratio.item(),
                        'w_expert:', w_e_expert.item(),
                        'w_offline:', w_e_offline.item())

            wandb.log(loss_result, step=total_it)

                # if total_it % (self.evaluate_freq * 2) == 0:
                    # self.save_parameters(evaluate_reward)
            self.total_it += 1

    def pretrain_discriminator(self):
        # Train discriminator
        if self.disc_type == 'learned':
            dataset_expert = torch.utils.data.TensorDataset(torch.FloatTensor(self.replay_buffer.expert_input_for_d))
            dataset_offline = torch.utils.data.TensorDataset(torch.FloatTensor(self.replay_buffer.offline_input_for_d))
            
            pin_memory = True
            expert_loader = torch.utils.data.DataLoader(dataset_expert, sampler=None, batch_size=256, shuffle=True, pin_memory=pin_memory, drop_last=True)
            offline_loader = torch.utils.data.DataLoader(dataset_offline, sampler=None, batch_size=256, shuffle=True, pin_memory=pin_memory, drop_last=True)

            for i in tqdm(range(self.disc_iteration)):
                loss = self.discriminator.update(expert_loader, offline_loader)
                wandb.log({'disc_loss': loss}, step=i)

    def get_ratio(self, state, action):
        obs_for_disc = state
        if self.state:
            disc_input = obs_for_disc
            log_dD_dE = self.discriminator.predict_reward(disc_input)
        else:
            try:
                act_for_disc = action
                disc_input = torch.cat([obs_for_disc, act_for_disc], axis=1)
                log_dD_dE = self.discriminator.predict_reward(disc_input)
            except:
                log_dD_dE = self.discriminator.predict_reward_robomimic(state, action)
        return log_dD_dE

    def optim_reward(self, state, action, pre_reward, next_state, done, result={}):
        # advantage
        reward_res = self.r_net(state, action, pre_reward)
        reward_w_res = pre_reward + reward_res

        with torch.no_grad():
            v = self.v_net(state)
            next_v = self.v_net(next_state)
        adv = (reward_w_res + (1 - done) * self.gamma * next_v - v) / self.alpha
        # r loss
        if self.f == 'kl':
            # clip advantage to [min, max], prevent NAN or Inf
            w_e = (torch.exp(torch.clamp(adv, max=ADV_MAX)))  # numerical stable, like IQL does
        else:
            w_e = self.f_star_prime(adv)

        assert (not torch.isnan(adv).any()) or (not torch.isinf(adv).any)
        assert (not torch.isnan(w_e).any()) or (not torch.isinf(w_e).any)

        with torch.no_grad():
            log_dD_dE = - self.get_ratio(state, action)

        assert (not torch.isnan(log_dD_dE).any()) or (not torch.isinf(log_dD_dE).any)

        if self.f == 'kl':
            r_loss = torch.mean(w_e * (log_dD_dE + torch.log(w_e+1e-10)) - torch.log(torch.max(w_e)+1e-10).detach())
        else:
            r_loss = torch.mean(torch.exp(-log_dD_dE)*self.f_fn(w_e * torch.exp(log_dD_dE)))
        result.update({'r_loss': torch.mean(r_loss),
                       'r_mean': torch.mean(reward_w_res),
                       'r_res': torch.mean(reward_res),
                       'log_dD_dE': torch.mean(log_dD_dE)})

        self.r_optim.zero_grad()
        r_loss.backward()
        self.r_optim.step()
        self.r_lr_schedule.step()
        return result, reward_w_res

    def get_V_policy_loss(self, init_state, state, action, reward, next_state, done, result={}):
        """
        train the V function
        return: v loss information
        """
        # advantage
        with torch.no_grad():
            reward_res = self.r_net(state, action, reward)
            reward_w_res = reward + reward_res

        initial_v = self.v_net(init_state)
        v = self.v_net(state)
        next_v = self.v_net(next_state)
        adv = (reward_w_res + (1 - done) * self.gamma * next_v - v) / self.alpha
        # adv =   # clip advantage to [min, max], prevent NAN or Inf

        assert (not torch.isnan(adv).any()) or (not torch.isinf(adv).any)

        # V loss
        v_loss0 = (1 - self.gamma) * initial_v
        if self.f == 'kl':
            v_loss1 = self.alpha * torch.log(torch.mean(torch.exp(torch.clamp(adv, max=ADV_MAX)))+1e-10)
        else:
            v_loss1 = self.alpha * torch.mean(self.f_star(adv))

        v_loss = v_loss0 + v_loss1
        v_loss = torch.mean(v_loss)

        result.update({
            'v_loss0': torch.mean(v_loss0),
            'v_loss1': torch.mean(v_loss1),
            'v_loss': v_loss,
        })

        # importance sampling ratio
        if self.f == 'kl':
            w_e = (torch.exp(adv).detach()).clamp(max=100) + 1e-20  # numerical stable, like IQL does
        else:
            w_e = self.f_star_prime(adv).detach() + 1e-20

        assert (not torch.isnan(w_e).any()) or (not torch.isinf(w_e).any)

        w_e = w_e / torch.sum(w_e)  # self-normalize IS

        assert (not torch.isnan(w_e).any()) or (not torch.isinf(w_e).any)

        if not self.actor_deterministic:
            # entropy
            _, log_pi, _ = self.actor_net(state)
            negative_entropy_loss = torch.mean(log_pi)

            # weighted bc
            action_log_prob = self.actor_net.get_log_density(state, action)
            policy_loss = -torch.mean(w_e.detach() * action_log_prob)

            result.update({'negative_entropy_loss': negative_entropy_loss})

            if self.use_policy_entropy_constraint:
                ent_coeff = torch.exp(self.log_ent_coeff).squeeze(0)
                policy_loss += ent_coeff * negative_entropy_loss

                ent_coeff_loss = -self.log_ent_coeff * (log_pi + self.target_entropy).detach()

                result.update({
                    'ent_coeff_loss': torch.mean(ent_coeff_loss),
                    'ent_coeff': ent_coeff,
                })
        else:
            # deterministic policy
            action_pi = self.actor_net(state)
            assert (not torch.isnan(action_pi).any()) or (not torch.isinf(action_pi).any)
            bc_losses = torch.sum((action_pi - action)**2, dim=1, keepdim=True)
            policy_loss = torch.mean(w_e.detach() * bc_losses)
            result.update({'bc_loss': torch.mean(bc_losses)})

        result.update({
            'w_e': w_e,
            'policy_loss': policy_loss})

        return result

    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        state = self.env.reset()
        ep_rews = 0.
        success_count = 0.

        # repeat multiple trials
        for i in range(self.evaluate_episodes):
            if self.env_name in ROBOMIMIC:
                horizon_length = 400
                success = { k: False for k in self.env.is_success() } # success metrics
            else:
                horizon_length = 1000
            
            # reset, start a new trial
            state = self.env.reset()
            # for each trial, perform horizon_length steps
            for steps in range(horizon_length):
                # transfer robomimic state to gym numpy state
                if self.env_name in ROBOMIMIC:
                    state = transfer_state(state, self.obs_key)

                # normalize state to N(0, 1)
                if self.standardize_obs:
                    state = (state - self.replay_buffer.dataset_statistics['observation_mean']) / (
                            self.replay_buffer.dataset_statistics['observation_std'] + 1e-5)
                
                # add absorbing state
                if self.absorbing_state:
                    state = np.append(state, 0)
                
                if self.env_name not in ROBOMIMIC:
                    state = state.squeeze()

                # use deterministic actor_net or stochastic actor net
                if self.actor_deterministic:
                    if self.env_name in ROBOMIMIC:
                        action = self.actor_net(state, train=False).squeeze(0).cpu().detach().numpy()
                    else:
                        action = self.actor_net(state, train=False).cpu().detach().numpy()
                else:
                    action = self.actor_net.deterministic_action(state).cpu().detach().numpy()

                # env step
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward

                # check whether done or success, if so reset, break and start a new episode
                if self.env_name not in ROBOMIMIC:
                    if done:
                        break
                else:
                    cur_success_metrics = self.env.is_success()
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]
                    if done or success["task"]:
                        success_count += 1
                        break
                
                # render
                if self.render:
                    if self.env_name in ROBOMIMIC:
                        self.env.render(mode='human')
                    else:
                        self.env.render()


        # episode reward calculation
        if self.env_name not in ROBOMIMIC:
            ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100 / self.evaluate_episodes
        else:
            ep_rews = success_count * 100 / self.evaluate_episodes
        return ep_rews

    def obs_to_cuda(self, s):
        obs_dict = {key: s[key].to(self.device) for key in s}
        return obs_dict

    def save_parameters(self, reward=0):
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}"
        os.makedirs(logdir_name)
        r_logdir_name = f"{logdir_name}/{self.env_name}-r.pth"
        torch.save(self.r_net.state_dict(), r_logdir_name)
