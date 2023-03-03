import d4rl 
import gym 
import numpy as np
import pickle 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from tqdm import tqdm 
from Network.Actor_Critic_net import ResNet18Conv, SpatialSoftmax, VisualCore
from torch.autograd import Variable
from robomimic.models.base_nets import CropRandomizer

ROBOMIMIC = ['lift', 'can', 'square', 'tool_hang', 'transport']

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/gail.py
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, device='cuda:0'):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=3e-4)

    def compute_grad_pen(self,
                         expert_state,
                         offline_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for expert_state, offline_state in zip(expert_loader, offline_loader):

            expert_state = expert_state[0].to(self.device)
            offline_state = offline_state[0][:expert_state.shape[0]].to(self.device)

            policy_d = self.trunk(offline_state)
            expert_d = self.trunk(expert_state)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, offline_state)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self.trunk(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 


class Discriminator_SA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, use_camera=False, env_name=None, device='cuda:0'):
        super(Discriminator_SA, self).__init__()

        self.env_name = env_name
        self.use_camera = use_camera
        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim
        state_hidden_dim = hidden_dim if action_dim == 0 else int(hidden_dim/2) 
        self.state_trunk = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim), nn.Tanh()).to(device)
        action_trunk_input_dim = 1 if action_dim == 0 else action_dim 
        self.action_trunk = nn.Sequential(
            nn.Linear(action_trunk_input_dim, int(hidden_dim/2)), nn.Tanh()).to(device)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.state_trunk.train()
        self.action_trunk.train()
        self.trunk.train()

        if use_camera:
            self.visualcore_agentview = self.get_one_visual_encoder()
            self.visualcore_wristview = self.get_one_visual_encoder()
            self.crop_randomizer = CropRandomizer([3, 84, 84], 76, 76)
            # self.visualcore_agentview.eval()
            # self.visualcore_wristview.eval()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.01)  # TODO only optimize the trunk parameter？
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.lr_schedule = CosineAnnealingLR(self.optimizer, 1000)
        else:
            self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=1e-3)  # TODO only optimize the trunk parameter？        

    def forward(self, input):
        if input.shape[1] == self.state_dim:
            h = self.state_trunk(input)
            h = self.trunk(h)
        else:
            state = input[:, :self.state_dim]
            action = input[:, self.state_dim:]
            h_state = self.state_trunk(state)
            h_action = self.action_trunk(action)
            h = torch.cat([h_state, h_action], axis=1)
            h = self.trunk(h)
        return h 

    def compute_grad_pen_image(self, expert_state, offline_state, lambda_=20):
        alpha_lowdim = torch.rand(expert_state['obs']['robot0_eef_pos'].shape[0], 1).to(self.device)
        alpha_image = alpha_lowdim.unsqueeze(1).unsqueeze(1)

        expert_agentview = expert_state['obs']['agentview_image'].squeeze(1)
        offline_agentview = offline_state['obs']['agentview_image'].squeeze(1)
        expert_wristview = expert_state['obs']['robot0_eye_in_hand_image'].squeeze(1)
        offline_wristview = offline_state['obs']['robot0_eye_in_hand_image'].squeeze(1)

        expert_a = expert_state['actions'].squeeze(1)
        offline_a = offline_state['actions'].squeeze(1)

        expert_lowdim_s = []
        offline_lowdim_s = []
        for key in expert_state['obs']:
            if key != 'robot0_eye_in_hand_image' and key != 'agentview_image':
                expert_lowdim_s.append(expert_state['obs'][key].squeeze(1))
        for key in offline_state['obs']:
            if key != 'robot0_eye_in_hand_image' and key != 'agentview_image':
                offline_lowdim_s.append(offline_state['obs'][key].squeeze(1))

        expert_lowdim_s = torch.concat(expert_lowdim_s, dim=1)
        offline_lowdim_s = torch.concat(offline_lowdim_s, dim=1)

        alpha_image = alpha_image.expand_as(expert_agentview)
        alpha_lowdim_s = alpha_lowdim.expand_as(expert_lowdim_s)
        alpha_a = alpha_lowdim.expand_as(expert_a)

        mix_lowdim = alpha_lowdim_s * expert_lowdim_s + (1 - alpha_lowdim_s) * offline_lowdim_s
        mix_a = alpha_a * expert_a + (1 - alpha_a) * offline_a
        mix_wristview = alpha_image * expert_wristview + (1 - alpha_image) * offline_wristview
        mix_agentview = alpha_image * expert_agentview + (1 - alpha_image) * offline_agentview

        mix_lowdim = Variable(mix_lowdim, requires_grad=True)
        mix_a = Variable(mix_a, requires_grad=True)
        mix_wristview = Variable(mix_wristview, requires_grad=True)
        mix_agentview = Variable(mix_agentview, requires_grad=True)

        mix_wrist_embed = self.vision_embedding(mix_wristview.to(self.device), embedding_type='wrist')
        mix_agent_embed = self.vision_embedding(mix_agentview.to(self.device), embedding_type='agent')

        mixup_data = torch.concat([mix_lowdim.to(self.device), mix_wrist_embed, mix_agent_embed, mix_a.to(self.device)], dim=1)

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)

        grad_lowdim = autograd.grad(
                        outputs=disc,
                        inputs=mix_lowdim,
                        grad_outputs=ones,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]

        grad_a = autograd.grad(
                        outputs=disc,
                        inputs=mix_a,
                        grad_outputs=ones,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]

        grad_wrist_image = autograd.grad(
                outputs=disc,
                inputs=mix_wristview,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_wrist_image = grad_wrist_image.reshape(expert_state['obs']['robot0_eef_pos'].shape[0], -1)

        grad_agent_image = autograd.grad(
                outputs=disc,
                inputs=mix_agentview,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_agent_image = grad_agent_image.reshape(expert_state['obs']['robot0_eef_pos'].shape[0], -1)

        grad = torch.cat([grad_lowdim, grad_wrist_image, grad_agent_image, grad_a], dim=1)

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def compute_grad_pen(self,
                         expert_state,
                         offline_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

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
    
    def get_embedded_image_state(self, dict_data, action=None):
        # self.visualcore_agentview.eval()
        # self.visualcore_wristview.eval()
        self.train()
        try:
            dict_obs = dict_data['obs']
        except:
            dict_obs = dict_data
        if action is None:
            a = dict_data['actions'].squeeze(1).to(self.device)
        else:
            a = action

        embed_state = []
        for key in dict_obs:
            if key == 'robot0_eye_in_hand_image':
                wristview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1)
                # wristview_image = dict_obs[key].squeeze(1).to(self.device)
                wrist_embed = self.vision_embedding(wristview_image, embedding_type='wrist')
                embed_state.append(wrist_embed)
            elif key == 'agentview_image':
                agentview_image = self.crop_randomizer.forward_in(dict_obs[key]).squeeze(1)
                # agentview_image = dict_obs[key].squeeze(1).to(self.device)
                agent_embed = self.vision_embedding(agentview_image, embedding_type='agent')
                embed_state.append(agent_embed)
            else:
                embed_state.append(dict_obs[key].squeeze(1))

        embed_state = torch.concat(embed_state, dim=1)
        return torch.concat([embed_state, a], dim=1)

    def init_loader(self, expert_loader, offline_loader):
        discriminator_loader = zip(expert_loader, offline_loader)
        self.discriminator_loader = discriminator_loader
        self.discriminator_loader_iter = iter(discriminator_loader)

    def init_test_loader(self, expert_loader, offline_loader):
        discriminator_loader = zip(expert_loader, offline_loader)
        self.test_discriminator_loader = discriminator_loader
        self.test_discriminator_loader_iter = iter(discriminator_loader)

    def obs_to_cuda(self, s):
        for key in s:
            if type(s[key]) is dict:
                for keykey in s[key]:
                    s[key][keykey] = s[key][keykey].to(self.device)
            else:
                s[key] = s[key].to(self.device)
        return s
        
    def update_image(self):
        # self.visualcore_agentview.eval()
        # self.visualcore_wristview.eval()
        self.train()

        expert_state, offline_state = next(self.discriminator_loader_iter)

        expert_state = self.obs_to_cuda(expert_state)
        offline_state = self.obs_to_cuda(offline_state)

        if self.use_camera:  # image observation
            expert_state_ = self.get_embedded_image_state(expert_state)
            offline_state_ = self.get_embedded_image_state(offline_state)
        else:
            expert_state_ = expert_state[0].to(self.device)
            offline_state_ = offline_state[0][:expert_state.shape[0]].to(self.device)

        policy_d = self(offline_state_)
        expert_d = self(expert_state_)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(self.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(self.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen_image(expert_state, offline_state)

        loss = (gail_loss + grad_pen).item()

        self.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        self.optimizer.step()
        self.lr_schedule.step()

        return loss


    def test(self):
        self.eval()

        expert_state, offline_state = next(self.test_discriminator_loader_iter)

        if self.use_camera:  # image observation
            expert_state_ = self.get_embedded_image_state(expert_state)
            offline_state_ = self.get_embedded_image_state(offline_state)
        else:
            expert_state_ = expert_state[0].to(self.device)
            offline_state_ = offline_state[0][:expert_state.shape[0]].to(self.device)

        policy_d = self(offline_state_)
        expert_d = self(expert_state_)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(self.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(self.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen_image(expert_state, offline_state)

        return (gail_loss + grad_pen).item()
        

    def update(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for expert_state, offline_state in zip(expert_loader, offline_loader):
            if self.env_name in ROBOMIMIC and self.use_camera:  # robomimic observation
                expert_state = self.get_embedded_image_state(expert_state)
                offline_state = self.get_embedded_image_state(offline_state)
            else:
                expert_state = expert_state[0].to(self.device)
                offline_state = offline_state[0][:expert_state.shape[0]].to(self.device)

            policy_d = self(offline_state)
            expert_d = self(expert_state)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, offline_state)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def update2(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for offline_state in tqdm(offline_loader):
            for expert_state in expert_loader:
                batch_size = min(offline_state[0].shape[0], expert_state[0].shape[0])
                offline_state = offline_state[0][:batch_size].to(self.device)
                expert_state = expert_state[0][:batch_size].to(self.device)
                policy_d = self(offline_state)
                expert_d = self(expert_state)

                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(self.device))
                policy_loss = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(self.device))

                gail_loss = expert_loss + policy_loss
                grad_pen = self.compute_grad_pen(expert_state, offline_state)

                loss += (gail_loss + grad_pen).item()
                n += 1

                self.optimizer.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer.step()
                # force just once in the inner-loop 
                break  
        return loss / n

    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 
    
    def predict_reward_robomimic(self, state, action):
        with torch.no_grad():
            self.eval()
            state = self.get_embedded_image_state(state, action)
            d = self(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 
