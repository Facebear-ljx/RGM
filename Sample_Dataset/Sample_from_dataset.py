import numpy as np
import torch
import collections

from robomimic.utils.dataset import SequenceDataset, ObsUtils
from torch.utils.data import DataLoader

# Used for robomimic
obs_dict = {
    "low_dim": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object"
    ],
    "rgb": [],
    "depth": [],
    "scan": []
}

obs_image_dict = {
    "low_dim": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos"
        # "object"
    ],
    "rgb": [                   
        "robot0_eye_in_hand_image",
        "agentview_image"
    ],
    "depth": [],
    "scan": []
}

obs_keys=(                      # observations we want to appear in batches
    "robot0_eef_pos", 
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object"
)

obs_image_keys=(                      # observations we want to appear in batches
    "robot0_eef_pos", 
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    # "object",
    "robot0_eye_in_hand_image",
    "agentview_image"
)

ROBOMIMIC = ['lift', 'can', 'square', 'tool_hang', 'transport']


def get_dataset_from_robomimic(dataset_path, num_trajectories=None, filter_by_attribute=None):
    # Initialize ObsUtils
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_dict)
    
    # Instance of SequenceDataset
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,              # observations we want to appear in batches
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        seq_length=1,                   # length-10 temporal sequences
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        hdf5_cache_mode=None,           # cache dataset in memory to avoid repeated file i/o
        hdf5_normalize_obs=False,
        filter_by_attribute=filter_by_attribute,       # can optionally provide a filter key here
    )
    train_sampler = dataset.get_dataset_sampler()

    train_loader = DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # change to D4RL dataset
    data_loader_iter = iter(train_loader)
    if num_trajectories is None:
        num_trajectories = len(train_loader)
    obs, new_obs, action, reward, done_bool, timeouts = [], [], [], [], [], []
    demo_start_indices = dataset._demo_id_to_start_indices.values()
    for i in range(num_trajectories):
        batch = next(data_loader_iter)
        action.append(np.array(batch['actions'].view(-1)))
        reward.append(np.array(batch['rewards'].view(-1)))
        done_bool.append(np.array(batch['dones'].view(-1)))

        # state concante
        s = batch['obs']
        next_s = batch['next_obs']
        obs_, next_obs_ = [], []
        for key in obs_keys:
            obs_.append(s[key].view(-1))
            next_obs_.append(next_s[key].view(-1))
        obs.append(np.array(torch.cat(obs_, dim=0)))
        new_obs.append(np.array(torch.cat(next_obs_, dim=0)))

        # time out indices
        if i < num_trajectories - 1:
            timeouts.append(((i+1) in demo_start_indices) + 0)
        else:
            timeouts.append(1)

    dataset_d4rltype = {
        'observations': np.array(obs, dtype=np.float32),
        'actions': np.array(action, dtype=np.float32),
        'next_observations': np.array(new_obs, dtype=np.float32),
        'rewards': np.array(reward, dtype=np.float32),
        'terminals': np.array(done_bool, dtype=np.float32),
        'timeouts': np.array(timeouts, dtype=np.float32)
    }
    return dataset_d4rltype


class DICEReplayBuffer(object):
    def __init__(self, 
                 env,
                 env_name,
                 expert_env,
                 num_expert_traj=200,
                 batch_size=256,
                 num_offline_traj=2000,
                 state=True,
                 standardize_observation=True,
                 absorbing_state=True,
                 standardize_reward=True,
                 reverse=False,
                 mh=None,
                 reward_type='P',
                 robomimic_data_path=None,
                 device='cpu'):
        
        self.env=env
        self.env_name=env_name
        self.expert_env=expert_env
        self.num_expert_traj=num_expert_traj
        self.num_offline_traj=num_offline_traj
        self.state=state
        self.standardize_observation=standardize_observation
        self.absorbing_state=absorbing_state
        self.standardize_reward=standardize_reward
        self.device=device
        self.batch_size=batch_size
        self.mh=mh
        self.reward_type=reward_type
        self.robomimic_data_path=robomimic_data_path

        if self.env_name in ROBOMIMIC:
            self.obs_keys = obs_keys
        
    def get_expert_traj(self):
        if 'antmaze' not in self.env_name:
            if self.env_name not in ROBOMIMIC:
                self.expert_dataset = self.expert_env.get_dataset()
            else:
                self.expert_dataset = get_dataset_from_robomimic(dataset_path=f'{self.robomimic_data_path}/{self.env_name}/ph/low_dim.hdf5')
            self.traj_iterator = self.sequence_dataset(self.expert_env, self.expert_dataset)
            self.expert_traj = next(self.traj_iterator)
        else:
            data = np.genfromtxt('data/Antmaze_medium_expert_s_l.csv', dtype=float, delimiter=',')
            action = np.genfromtxt('data/Antmaze_medium_expert_a_l.csv', dtype=float, delimiter=',')
            state = data[:len(data)-1, :]
            action = action[:len(data)-1, :]
            next_state = data[1:len(data), :]
            done = np.zeros(state.shape[0])
            r = np.zeros_like(done)
            for i in range(len(state)-2):
                state_0 = state[i][0]
                state_1 = state[i+1][0]
                if abs(state_1 - state_0) > 15:
                    done[i] = 1
                    r[i] = 1
            done[state.shape[0]-1] = 1
            print(len(data))
            self.expert_traj = {'observations': state,
                                'actions': action,
                                'rewards': r,
                                'next_observations': next_state,
                                'terminals': done}


    def get_offline_traj(self):
        if self.env_name in ROBOMIMIC:
            data_type='mg/low_dim_sparse'
            filter_by_arrtibute = None
            # use robomimic SequenceDataset to process image data, save memory
            offline_dataset = get_dataset_from_robomimic(dataset_path=f'{self.robomimic_data_path}/{self.env_name}/{data_type}.hdf5', filter_by_attribute=filter_by_arrtibute)
            # this function will load all data into cpu, may cause memory exposion when processing image data
        else:
            offline_dataset, self.expert_dataset = None, None

        if self.num_expert_traj == 0:
            self.initial_obs_dataset, self.dataset, self.dataset_statistics = self.dice_dataset(self.env,
                                                                                                self.standardize_observation,
                                                                                                self.absorbing_state,
                                                                                                self.standardize_reward,
                                                                                                dataset=offline_dataset)
        else:
            self.initial_obs_dataset, self.dataset, self.dataset_statistics = self.dice_combined_dataset(self.expert_env,
                                                                                                         self.env,
                                                                                                         self.num_expert_traj,
                                                                                                         self.num_offline_traj,
                                                                                                         standardize_observation=self.standardize_observation,
                                                                                                         absorbing_state=self.absorbing_state,
                                                                                                         standardize_reward=self.standardize_reward,
                                                                                                         offline_dataset=offline_dataset,
                                                                                                         expert_dataset=self.expert_dataset)
    
    
    def normalize_expert(self):
        if self.env_name in ROBOMIMIC:
            if self.standardize_observation:
                expert_obs_dim = self.expert_dataset['observations'].shape[1]
                self.expert_dataset['observations'] = (self.expert_dataset['observations'] - self.dataset_statistics['observation_mean'][:expert_obs_dim]) / (self.dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
                if 'next_observations' in self.expert_dataset:
                    self.expert_dataset['next_observations'] = (self.expert_dataset['next_observations'] - self.dataset_statistics['observation_mean']) / (self.dataset_statistics['observation_std'] + 1e-10)
            if self.absorbing_state:
                self.expert_dataset = self.add_absorbing_state(self.expert_dataset)
            if self.env_name not in ROBOMIMIC:
                self.target_entropy = -np.prod(self.env.action_space.shape)
            else:
                self.target_entropy = -np.array(7)
        else:
            if self.standardize_observation:
                expert_obs_dim = self.expert_traj['observations'].shape[1]
                self.expert_traj['observations'] = (self.expert_traj['observations'] - self.dataset_statistics['observation_mean'][:expert_obs_dim]) / (self.dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
                if 'next_observations' in self.expert_traj:
                    self.expert_traj['next_observations'] = (self.expert_traj['next_observations'] - self.dataset_statistics['observation_mean']) / (self.dataset_statistics['observation_std'] + 1e-10)
            if self.absorbing_state:
                self.expert_traj = self.add_absorbing_state(self.expert_traj)
            if self.env_name not in ROBOMIMIC:
                self.target_entropy = -np.prod(self.env.action_space.shape)
            else:
                self.target_entropy = -np.array(7)
    
    
    def get_discriminator_dataset(self):
        # Create inputs for the discriminator
        state_dim = self.dataset_statistics['observation_dim'] + 1 if self.absorbing_state else self.dataset_statistics['observation_dim']
        action_dim = 0 if self.state else self.dataset_statistics['action_dim']
        disc_cutoff = state_dim

        if self.state:
            self.expert_input_for_d = self.expert_traj['observations'][:, :disc_cutoff]
            self.offline_input_for_d = self.dataset['observations'][:, :disc_cutoff]
        else:
            if self.env_name not in ROBOMIMIC:
                # used for d4rl dataset
                self.expert_state_for_d = self.expert_traj['observations'][:, :disc_cutoff]
                self.expert_action_for_d = self.expert_traj['actions'][:, :action_dim]
                self.offline_state_for_d = self.dataset['observations'][:, :disc_cutoff]
                self.offline_action_for_d = self.dataset['actions'][:, :action_dim]

                self.expert_input_for_d = np.concatenate([self.expert_state_for_d, self.expert_action_for_d], axis=1)
                self.offline_input_for_d = np.concatenate([self.offline_state_for_d, self.offline_action_for_d], axis=1)
            else:
                # used for robomimic low_dim dataset
                self.expert_state_for_d = self.expert_dataset['observations'][:, :disc_cutoff]
                self.expert_action_for_d = self.expert_dataset['actions'][:, :action_dim]
                self.offline_state_for_d = self.dataset['observations'][:, :disc_cutoff]
                self.offline_action_for_d = self.dataset['actions'][:, :action_dim]
                self.expert_input_for_d = np.concatenate([self.expert_state_for_d, self.expert_action_for_d], axis=1)
                self.offline_input_for_d = np.concatenate([self.offline_state_for_d, self.offline_action_for_d], axis=1)
    
    def prepare_dicedataset(self):
        self.get_expert_traj()
        self.get_offline_traj()
        self.normalize_expert()
        self.get_discriminator_dataset()
        return self.target_entropy 
    
    
    def sequence_dataset(self, env, dataset):
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)
        
        fields = ['actions', 'observations', 'rewards', 'terminals']
        if 'infos/qpos' in dataset:
            fields.append('infos/qpos')
            fields.append('infos/qvel')
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True
        
        episode_step = 0
        if 'next_observations' in dataset.keys():
            fields.append('next_observations')

        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)

            for k in fields:
                data_[k].append(dataset[k][i])

            if self.env_name in ROBOMIMIC:
                if final_timestep:
                    episode_step = 0
                    self.episode_data = {}
                    for k in data_:
                        self.episode_data[k] = np.array(data_[k])

                    yield self.episode_data 
                    data_ = collections.defaultdict(list)
            else:
                if done_bool or final_timestep:
                    episode_step = 0
                    self.episode_data = {}
                    for k in data_:
                        self.episode_data[k] = np.array(data_[k])

                    yield self.episode_data 
                    data_ = collections.defaultdict(list)   

            episode_step += 1


    def dice_robomimic_dataset(self, env, standardize_observation=True, absorbing_state=True, standardize_reward=False, dataset=None):
        """
        env: robomimic environment

        output: init_dataset -> list; dataset -> Dataset
        """
        if dataset is None:
            raise NotImplementedError

        start_indices = dataset._demo_id_to_start_indices.values()
        init_dataset = []

        for i in start_indices:
            init_dataset.append(dataset[i])   # list

        pin_memory=True

        obs_statistic = None
        if self.standardize_observation:
            obs_statistic = dataset.get_obs_normalization_stats()

        self.init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True)
        self.dice_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True)

        self.robomimic_dataloader = zip(self.init_loader, self.dice_dataset_loader)
        return init_dataset, dataset, obs_statistic
        

    def dice_dataset(self, env, standardize_observation=True, absorbing_state=True, standardize_reward=True, dataset=None):
        """
        env: d4rl environment
        """
        if dataset is None:
            dataset = env.get_dataset()
        N = dataset['rewards'].shape[0]
        initial_obs_, obs_, next_obs_, action_, reward_, done_, expert_ = [], [], [], [], [], [], []

        use_timeouts = ('timeouts' in dataset)

        episode_step = 0
        reverse_current_traj = False
        for i in range(N-1):
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
            if is_final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue

            if episode_step == 0:
                initial_obs_.append(obs)

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            expert_.append(bool(0)) # assume not expert
            episode_step += 1

            if self.env_name in ROBOMIMIC:
                if is_final_timestep:
                    episode_step = 0
            else:
                if done_bool or is_final_timestep:
                    episode_step = 0

        initial_obs_dataset = {
            'initial_observations': np.array(initial_obs_, dtype=np.float32)
        }
        dataset = {
            'observations': np.array(obs_, dtype=np.float32),
            'actions': np.array(action_, dtype=np.float32),
            'next_observations': np.array(next_obs_, dtype=np.float32),
            'rewards': np.array(reward_, dtype=np.float32),
            'terminals': np.array(done_, dtype=np.float32),
            'experts': np.array(expert_, dtype=np.float32)
        }
        
        if self.reward_type == 'P':
            reward_noise = np.random.uniform(0, 1, len(reward_))
            thre = 0.5
            dataset['rewards'][reward_noise > thre] *= -1
        elif self.reward_type == 'C':
            dataset['rewards'] *= -1
            
        dataset_statistics = {
            'observation_mean': np.mean(dataset['observations'], axis=0),
            'observation_std': np.std(dataset['observations'], axis=0),
            'reward_mean': np.mean(dataset['rewards']),
            'reward_std': np.std(dataset['rewards']),
            'N_initial_observations': len(initial_obs_),
            'N': len(obs_),
            'observation_dim': dataset['observations'].shape[-1],
            'action_dim': dataset['actions'].shape[-1]
        }

        if standardize_observation:
            initial_obs_dataset['initial_observations'] = (initial_obs_dataset['initial_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
            dataset['observations'] = (dataset['observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
            dataset['next_observations'] = (dataset['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        if standardize_reward:
            dataset['rewards'] = (dataset['rewards'] - dataset_statistics['reward_mean']) / (dataset_statistics['reward_std'] + 1e-10)

        if absorbing_state:
            # add additional dimension to observations to deal with absorbing state
            initial_obs_dataset['initial_observations'] = np.concatenate((initial_obs_dataset['initial_observations'], np.zeros((dataset_statistics['N_initial_observations'], 1))), axis=1).astype(np.float32)
            dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
            dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
            terminal_indices = np.where(dataset['terminals'])[0]
            absorbing_state = np.eye(dataset_statistics['observation_dim'] + 1)[-1].astype(np.float32)
            dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
                list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
            for terminal_idx in terminal_indices:
                dataset['next_observations'][terminal_idx] = absorbing_state
                dataset['observations'].append(absorbing_state)
                dataset['actions'].append(dataset['actions'][terminal_idx])
                dataset['rewards'].append(0)
                dataset['next_observations'].append(absorbing_state)
                dataset['terminals'].append(1)

            dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
                np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
                np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)

        return initial_obs_dataset, dataset, dataset_statistics


    def dice_combined_dataset(self, expert_env, env, num_expert_traj=2000, num_offline_traj=2000, expert_dataset=None, offline_dataset=None,
                                standardize_observation=True, absorbing_state=True, standardize_reward=True, reverse=False):
        """
        env: d4rl environment
        """
        initial_obs_, obs_, next_obs_, action_, reward_, done_, expert_ = [], [], [], [], [], [], []

        def add_data(env, num_traj, dataset=None, expert_data=False):
            if dataset is None:
                dataset = env.get_dataset()
            N = dataset['rewards'].shape[0]
            use_timeouts = ('timeouts' in dataset)
            traj_count = 0
            episode_step = 0
            reverse_current_traj = 0
            for i in range(N-1):
                # only use this condition when num_traj < 2000
                if num_traj != 2000 and traj_count == num_traj:
                    break
                obs = dataset['observations'][i].astype(np.float32)
                new_obs = dataset['observations'][i+1].astype(np.float32)
                action = dataset['actions'][i].astype(np.float32)
                reward = dataset['rewards'][i].astype(np.float32)
                done_bool = bool(dataset['terminals'][i])

                is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
                if is_final_timestep:
                    # Skip this transition and don't apply terminals on the last step of an episode
                    traj_count += 1
                    episode_step = 0
                    reverse_current_traj = not reverse_current_traj
                    continue

                if episode_step == 0:
                    initial_obs_.append(obs)

                obs_.append(obs)
                next_obs_.append(new_obs)
                action_.append(action)
                reward_.append(reward)
                done_.append(done_bool)
                expert_.append(expert_data)
                episode_step += 1
                
                if self.env_name in ROBOMIMIC:
                    if is_final_timestep:
                        traj_count += 1
                        episode_step = 0
                        reverse_current_traj = not reverse_current_traj
                else:
                    if is_final_timestep or done_bool:
                        traj_count += 1
                        episode_step = 0
                        reverse_current_traj = not reverse_current_traj   
            a = 1

        add_data(expert_env, num_expert_traj, dataset=expert_dataset, expert_data=True)
        expert_size = len(obs_)
        print(f"Expert Traj {num_expert_traj}, Expert Size {expert_size}")
        add_data(env, num_offline_traj, dataset=offline_dataset, expert_data=False)
        offline_size = len(obs_) - expert_size 
        print(f"Offline Traj {num_offline_traj}, Offline Size {offline_size}")
        
        initial_obs_dataset = {
            'initial_observations': np.array(initial_obs_, dtype=np.float32)
        }
        dataset = {
            'observations': np.array(obs_, dtype=np.float32),
            'actions': np.array(action_, dtype=np.float32),
            'next_observations': np.array(next_obs_, dtype=np.float32),
            'rewards': np.array(reward_, dtype=np.float32),
            'terminals': np.array(done_, dtype=np.float32),
            'experts': np.array(expert_, dtype=np.float32)
        }
        
        if self.reward_type == 'P':
            reward_noise = np.random.uniform(0, 1, len(reward_))
            thre = 0.5
            dataset['rewards'][reward_noise > thre] *= -1
        elif self.reward_type == 'C':
            dataset['rewards'] *= -1
            
        dataset_statistics = {
            'observation_mean': np.mean(dataset['observations'], axis=0),
            'observation_std': np.std(dataset['observations'], axis=0),
            'reward_mean': np.mean(dataset['rewards']),
            'reward_std': np.std(dataset['rewards']),
            'N_initial_observations': len(initial_obs_),
            'N': len(obs_),
            'observation_dim': dataset['observations'].shape[-1],
            'action_dim': dataset['actions'].shape[-1]
        }

        if standardize_observation:
            initial_obs_dataset['initial_observations'] = (initial_obs_dataset['initial_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
            dataset['observations'] = (dataset['observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
            dataset['next_observations'] = (dataset['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        if standardize_reward:
            dataset['rewards'] = (dataset['rewards'] - dataset_statistics['reward_mean']) / (dataset_statistics['reward_std'] + 1e-10)

        if absorbing_state:
            # add additional dimension to observations to deal with absorbing state
            initial_obs_dataset['initial_observations'] = np.concatenate((initial_obs_dataset['initial_observations'], np.zeros((dataset_statistics['N_initial_observations'], 1))), axis=1).astype(np.float32)
            dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
            dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
            terminal_indices = np.where(dataset['terminals'])[0]
            absorbing_state = np.eye(dataset_statistics['observation_dim'] + 1)[-1].astype(np.float32)
            dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
                list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
            for terminal_idx in terminal_indices:
                dataset['next_observations'][terminal_idx] = absorbing_state
                dataset['observations'].append(absorbing_state)
                dataset['actions'].append(dataset['actions'][terminal_idx])
                dataset['rewards'].append(0)
                dataset['next_observations'].append(absorbing_state)
                dataset['terminals'].append(1)

            dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
                np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
                np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)

        return initial_obs_dataset, dataset, dataset_statistics


    def add_absorbing_state(self, dataset):
        N = dataset['observations'].shape[0]
        obs_dim = dataset['observations'].shape[1]
        dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((N, 1))), axis=1).astype(np.float32)
        dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((N, 1))), axis=1).astype(np.float32)
        terminal_indices = np.where(dataset['terminals'])[0]
        absorbing_state = np.eye(obs_dim + 1)[-1].astype(np.float32)
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
        for terminal_idx in terminal_indices:
            dataset['next_observations'][terminal_idx] = absorbing_state
            dataset['observations'].append(absorbing_state)
            dataset['actions'].append(dataset['actions'][terminal_idx])
            dataset['rewards'].append(0)
            dataset['next_observations'].append(absorbing_state)
            dataset['terminals'].append(1)

        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
            np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)
         
        return dataset   


    def _sample_minibatch(self, batch_size, reward_scale):
        initial_indices = np.random.randint(0, self.dataset_statistics['N_initial_observations'], batch_size)
        indices = np.random.randint(0, self.dataset_statistics['N'], batch_size)
        sampled_dataset = (
            self.initial_obs_dataset['initial_observations'][initial_indices],
            self.dataset['observations'][indices],
            self.dataset['actions'][indices],
            self.dataset['rewards'][indices] * reward_scale,
            self.dataset['next_observations'][indices],
            self.dataset['terminals'][indices],
            self.dataset['experts'][indices]
        )
        return tuple(map(torch.from_numpy, sampled_dataset))
    
    def sample_robomimic(self):
        try:
            init_batch, dice_batch = next(self.robomimic_dataloader_iter)
        except:
            self.robomimic_dataloader_iter = iter(self.robomimic_dataloader)
            init_batch, dice_batch = next(self.robomimic_dataloader_iter)
        return init_batch, dice_batch

    def init_dataloader(self):
        self.robomimic_dataloader = zip(self.init_loader, self.dice_dataset_loader)
        