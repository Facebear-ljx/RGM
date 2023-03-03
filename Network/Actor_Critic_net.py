from tkinter import N
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from torchvision import models as vision_models
from robomimic.models.base_nets import CropRandomizer
import math

MEAN_MIN = -7.24
MEAN_MAX = 7.24
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6
ROBOMIMIC = ['lift', 'can', 'square', 'tool_hang', 'transport']

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))


class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, action_space=None):
        """
        Stochastic Actor network, used for SAC, SBAC, BEAR, CQL
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.mean_min, self.mean_max = MEAN_MIN, MEAN_MAX
        self.logstd_min, self.logstd_max = LOG_STD_MIN, LOG_STD_MAX
        self.eps = EPS

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution

    def get_log_density(self, x, y):
        """
        calculate the log probability of the action conditioned on state
        :param x: state
        :param y: action
        :return: log(P(action|state))
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clip(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        """
        generate actions according to the state
        :param x: state
        :return: action
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action

    def get_action_multiple(self, x, num_samples=10):
        """
        used in BEAR
        :param x:
        :param num_samples:
        :return:
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = x.unsqueeze(0).repeat(num_samples, 1, 1).permute(1, 0, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        return torch.tanh(action), action

    def deterministic_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)
        return mu


class Actor_deterministic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, max_action=1., use_camera=False, env_name=None):
        """
        Deterministic Actor network, used for TD3, DDPG, BCQ, TD3_BC
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Actor_deterministic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.action = nn.Linear(num_hidden, num_action)
        self.max_action = max_action
        self.use_camera = use_camera
        self.env_name = env_name

        if self.use_camera:
            self.visualcore_agentview = self.get_one_visual_encoder()
            self.visualcore_wristview = self.get_one_visual_encoder()
            self.crop_randomizer = CropRandomizer([3, 84, 84], 76, 76)

    def forward(self, x, train=True):
        if train:
            self.train()
        else:
            self.eval()

        if self.env_name in ROBOMIMIC and self.use_camera:
            x = self.get_embedded_image_state(x)
        else:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float).to(self.device)

        a = F.relu(self.fc1(x))
        assert (not torch.isnan(a).any()) or (not torch.isinf(a).any)
        a = F.relu(self.fc2(a))
        assert (not torch.isnan(a).any()) or (not torch.isinf(a).any)
        a = self.action(a)
        assert (not torch.isnan(a).any()) or (not torch.isinf(a).any)
        return torch.tanh(a) * self.max_action
    
    def get_one_visual_encoder(self):   
        input_shape = [3, 76, 76]
        feature_dim = 64
        
        # backbone
        backbone = ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=False)
        feat_shape = backbone.output_shape(input_shape)
        net_list = [backbone]
        
        # pool
        pool = SpatialSoftmax(feat_shape, num_kp=32)
        feat_shape = pool.output_shape(feat_shape)
        net_list.append(pool)

        # flatten layer
        net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # linear layer
        linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dim)
        net_list.append(linear)

        return nn.Sequential(*net_list).to(self.device)

    def vision_embedding(self, image, embedding_type):
        if embedding_type == 'wrist':
            return self.visualcore_wristview(image)
        elif embedding_type == 'agent':
            return self.visualcore_agentview(image)
        else:
            raise NotImplementedError

    def get_embedded_image_state(self, dict_data):
        dict_obs = dict_data
        embed_state = []
        for key in dict_obs:
            if key == 'robot0_eye_in_hand_image':
                if len(dict_obs[key].shape) == 5:
                    wristview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                    # wristview_image = dict_obs[key].squeeze(1).to(self.device)
                else:
                    wristview_image = self.crop_randomizer.forward_in(dict_obs[key]).unsqueeze(0).to(self.device)
                    # wristview_image = dict_obs[key].unsqueeze(0).to(self.device)
                wrist_embed = self.vision_embedding(wristview_image, embedding_type='wrist')

                assert (not torch.isnan(wrist_embed).any()) or (not torch.isinf(wrist_embed).any)
                embed_state.append(wrist_embed)
            elif key == 'agentview_image':
                if len(dict_obs[key].shape) == 5:
                    agentview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                    # agentview_image = dict_obs[key].squeeze(1).to(self.device)
                else:
                    agentview_image = self.crop_randomizer.forward_in(dict_obs[key]).unsqueeze(0).to(self.device)
                    # agentview_image = dict_obs[key].unsqueeze(0).to(self.device)
                agent_embed = self.vision_embedding(agentview_image, embedding_type='agent')

                assert (not torch.isnan(agent_embed).any()) or (not torch.isinf(agent_embed).any)
                embed_state.append(agent_embed)
            else:
                if len(dict_obs[key].shape) == 3:
                    embed_state.append(dict_obs[key].squeeze(1).to(self.device))
                else:
                    embed_state.append(dict_obs[key].unsqueeze(0).to(self.device))

        return torch.concat(embed_state, dim=1)

class Double_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Double Q network, used for TD3_BC, BCQ
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Double_Critic, self).__init__()
        self.device = device

        # Q1 architecture
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(num_state + num_action, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)

    def forward(self, x, y):
        sa = torch.cat([x, y], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class V_critic(nn.Module):
    def __init__(self, num_state, num_hidden, device, use_camera=False, env_name=None):
        super(V_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)
        self.use_camera = use_camera
        self.env_name = env_name

        if self.use_camera:
            self.visualcore_agentview = self.get_one_visual_encoder()
            self.visualcore_wristview = self.get_one_visual_encoder()
            self.crop_randomizer = CropRandomizer([3, 84, 84], 76, 76)
        self.apply(weights_init_)
        
    def forward(self, x):
        if self.env_name in ROBOMIMIC and self.use_camera:
            x = self.get_embedded_image_state(x)
        else:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v

    def initialize(self):
        self.apply(weights_init_)

    def get_one_visual_encoder(self):
        input_shape = [3, 76, 76]
        feature_dim = 64
        
        # backbone
        backbone = ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=False)
        feat_shape = backbone.output_shape(input_shape)
        net_list = [backbone]
        
        # pool
        pool = SpatialSoftmax(feat_shape, num_kp=32)
        feat_shape = pool.output_shape(feat_shape)
        net_list.append(pool)

        # flatten layer
        net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # linear layer
        linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dim)
        net_list.append(linear)

        return nn.Sequential(*net_list).to(self.device)

    def vision_embedding(self, image, embedding_type):
        if embedding_type == 'wrist':
            return self.visualcore_wristview(image)
        elif embedding_type == 'agent':
            return self.visualcore_agentview(image)
        else:
            raise NotImplementedError

    def get_embedded_image_state(self, dict_data):
        self.train()
        
        dict_obs = dict_data
        embed_state = []
        for key in dict_obs:
            if key == 'robot0_eye_in_hand_image':
                # wristview_image = dict_obs[key].squeeze(1).to(self.device)  # [B, 1, C, H, W] -> [B, C, H, W]
                wristview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                wrist_embed = self.vision_embedding(wristview_image, embedding_type='wrist')  # [B, D]
                embed_state.append(wrist_embed)
            elif key == 'agentview_image':
                agentview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                # agentview_image = dict_obs[key].squeeze(1).to(self.device)  # [B, 1, C, H, W] -> [B, C, H, W]
                agent_embed = self.vision_embedding(agentview_image, embedding_type='agent')  # [B, D]
                embed_state.append(agent_embed)
            else:
                embed_state.append(dict_obs[key].squeeze(1).to(self.device))  # [B, 1, D] -> [B, D]

        return torch.concat(embed_state, dim=1)


class Q_critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(Q_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.q_value = nn.Linear(num_hidden, 1)

        self.apply(weights_init_)

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q_value(x)
        return q

    def initialize(self):
        self.apply(weights_init_)


class V_with_r(nn.Module):
    def __init__(self, num_state, num_hidden, device):
        super(V_with_r, self).__init__()
        self.device = device
        hidden_dim = int(num_hidden/2)
        self.fcs = nn.Linear(num_state, hidden_dim)
        self.fcr = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

        # self.apply(weights_init_)

    def forward(self, x, r):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r, dtype=torch.float).to(self.device)

        hs = self.fcs(x)
        hr = self.fcr(r)

        x = torch.cat([hs, hr], dim=1)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v


class Reward(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, res_scale, device, use_camera=False, env_name=None):
        super(Reward, self).__init__()
        self.device = device
        self.res_scale = res_scale
        self.fc_sar = nn.Linear(num_state + num_action + 1, num_hidden)
        self.fc_sa = nn.Linear(num_state + num_action, num_hidden)
        self.fc_s = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.reward = nn.Linear(num_hidden, 1)
        self.use_camera = use_camera
        self.env_name = env_name

        if self.use_camera:
            self.visualcore_agentview = self.get_one_visual_encoder()
            self.visualcore_wristview = self.get_one_visual_encoder()
            self.crop_randomizer = CropRandomizer([3, 84, 84], 76, 76)
        self.apply(weights_init_)

    def forward(self, x, y=None, r=None):
        if self.env_name in ROBOMIMIC and self.use_camera:
            x = self.get_embedded_image_state(x)
        else:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r, dtype=torch.float).to(self.device)
        if y is None:
            # r(s)
            x = F.relu(self.fc_s(x))
        else:
            if r is None:
                # r(s,a)
                x = torch.cat([x, y], dim=1)
                x = F.relu(self.fc_sa(x))
            else:
                # r(s,a,r)
                x = torch.cat([x, y, r], dim=1)
                x = F.relu(self.fc_sar(x))
        x = F.relu(self.fc2(x))
        r = self.reward(x)
        r = torch.tanh(r) * self.res_scale
        return r

    def get_one_visual_encoder(self):
        input_shape = [3, 76, 76]
        feature_dim = 64
        
        # backbone
        backbone = ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=False)
        feat_shape = backbone.output_shape(input_shape)
        net_list = [backbone]
        
        # pool
        pool = SpatialSoftmax(feat_shape, num_kp=32)
        feat_shape = pool.output_shape(feat_shape)
        net_list.append(pool)

        # flatten layer
        net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # linear layer
        linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dim)
        net_list.append(linear)

        return nn.Sequential(*net_list).to(self.device)

    def vision_embedding(self, image, embedding_type):
        if embedding_type == 'wrist':
            return self.visualcore_wristview(image)
        elif embedding_type == 'agent':
            return self.visualcore_agentview(image)
        else:
            raise NotImplementedError

    def get_embedded_image_state(self, dict_data):
        self.train()
        
        dict_obs = dict_data
        embed_state = []
        for key in dict_obs:
            if key == 'robot0_eye_in_hand_image':
                wristview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                wrist_embed = self.vision_embedding(wristview_image, embedding_type='wrist')  # [B, D]
                embed_state.append(wrist_embed)
            elif key == 'agentview_image':
                agentview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1).to(self.device)
                agent_embed = self.vision_embedding(agentview_image, embedding_type='agent')
                embed_state.append(agent_embed)
            else:
                embed_state.append(dict_obs[key].squeeze(1).to(self.device))

        return torch.concat(embed_state, dim=1)


class ResNet18Conv(nn.Module):  # TODO, 接入RGM算法内
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        # if input_coord_conv:
        #     net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # elif input_channel != 3:
        #     net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape=[512, 3, 3],
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class VisualCore(nn.Module):
    def __init__(
        self,
        input_shape,
        backbone_net=None, # Resnet18Conv
        pool_net=None,  # SpatialSoftmax
        flatten=True,
        feature_dimension=64,  # 64
    ):
        super(VisualCore, self).__init__(input_shape=input_shape)

        self.input_shape = input_shape
        self.flatten = flatten

        # Backbone
        net_list = [backbone_net]
        feat_shape = backbone_net.output_shape(input_shape)

        # Pooling
        if pool_net is not None:
            net_list.append(pool_net)
            feat_shape = pool_net.output_shape(input_shape)

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)