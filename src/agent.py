
import torch
import torch.optim as optim
import torch.functional as F
from torch import nn
from collections import deque
import random
import logging
import numpy as np
from typing import List
import os


from transition import Transition, State
from model import make_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory():
    '''replay memory with fixed capacity

    '''
    def __init__(self, capacity):
        '''init a replay buffer with fixed capacity

        Args:
            capacity (int): capacity of replay memory
        '''
        self.memory = deque([], maxlen=capacity)

    def store(self, transition: Transition):
        '''store a new Transition into replay buffer

        Args:
            transition (Transition): transition to be added
        '''
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        '''sample a batch from replay buffer

        Args:
            batch_size (int): sample a batch

        Returns:
            _type_: List[Transition]
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, config):
        '''init the agent

        Args:
            config (Config): config file parsed from command args
        '''

        logging.info("Init Agent")
        self.state_dim = config.obs_height * config.obs_width
        self.obs_width = config.obs_width
        self.obs_height = config.obs_height

        self.action_dim = config.action_dim

        self.model_dir = config.model_dir

        self.batch_size = config.batch_size
        self.discount = config.discount

        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.lr_decay_every = config.lr_decay_every
        self.update_target_every = config.update_target_every

        self.ckpt_dir = os.path.join(config.model_dir, config.model_name)

        # Replay Buffer
        self.replay_buffer = ReplayMemory(capacity=config.replay_capacity)
        # Policy Net
        
        self.policy_net = make_model(config).to(device)
        if config.load_ckpt: 
            if os.path.exists(os.path.join(self.ckpt_dir, config.ckpt_name)):
                self.policy_net.load_state_dict(torch.load(os.path.join(self.ckpt_dir, config.ckpt_name)))
            else:
                logging.error(f"No checkpoint's name is {config.ckpt_name}!")

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
        self.DQN_criterion = nn.SmoothL1Loss()
        # Target Net
        self.target_net = make_model(config).to(device)


        ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints") 


        self.update_target_net()

        self.step = 0



    def update_target_net(self):
        '''update target network with policy network
        '''

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        logging.info("Target Network Updated")

    def store_transition(self, transition: Transition):
        '''store transition to replay buffer

        Args:
            transition (Transition):
        '''
        logging.debug("store new transition, current size: {}".format(len(self.replay_buffer)))
        self.replay_buffer.store(transition)
        
    def DQN_loss(self, state_q_values, next_state_q_values):
        '''Compute DQN loss

        Args:
            state_q_values (tensor): curr state's q values
            next_state_q_values (tensor): next state's q values
        Return:
            float: DQN Loss
        '''
        return self.DQN_criterion(state_q_values, next_state_q_values)

    
    def act(self, state: State):
        '''act on input state 

        Args:
            state (State): _description_

        Returns:
            numpy: predicted prob on input state
        '''
        state = (
            torch.from_numpy(state.image).float().to(device).unsqueeze(0) ,
            torch.from_numpy(state.pose_result['bbox']).to(device).flatten(),
            torch.from_numpy(state.pose_result['keypoints']).to(device).flatten()
        )

        self.policy_net.eval()
        with torch.no_grad():
            out = self.policy_net(state)
            out = out.cpu().squeeze(0).numpy()
        self.policy_net.train()
        
        return out

    def train_Q_network(self, update=True):
        '''train Q network for one step
            (Optional) Update the target net by policy net after given frequency

        Args:
            update (bool, optional): whether to update target model. Defaults to True.
        '''

        minibatch = self.replay_buffer.sample(self.batch_size)

        np.random.shuffle(minibatch)
        minibatch = minibatch
        # state batch
        state_batch = [data.state for data in minibatch]
        
        state_batch = (
            torch.stack(
                [torch.from_numpy(state.image)  for state in state_batch]
            ).float().to(device),
            torch.stack(
                [torch.from_numpy(state.pose_result['bbox']).flatten() for state in state_batch]
            ).to(device),
            torch.stack(
                [torch.from_numpy(state.pose_result['keypoints']).flatten() for state in state_batch]
            ).to(device)
        )
        logging.debug("state batch shape: {}".format(state_batch[0].shape))

        # action batch
        action_batch = torch.tensor(
            [data.action for data in minibatch], device=device
        ).long()

        logging.debug("action batch shape: {}".format(action_batch.shape))


        # next state batch
        next_state_batch = [data.next_state for data in minibatch]

        # reward batch
        reward_state_batch = torch.tensor(
            [data.reward for data in minibatch], device=device
        ).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        not_done_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state_batch)), 
            device=device,
            dtype=torch.bool
        )

        not_done_next_states = (
            torch.stack(
                [ torch.from_numpy(s.image)  for s in next_state_batch if s is not None]
            ).float().to(device),
            torch.stack(
                [torch.from_numpy(s.pose_result['bbox']).flatten() for s in next_state_batch if s is not None]
            ).to(device),
            torch.stack(
                [torch.from_numpy(s.pose_result['keypoints']).flatten() for s in next_state_batch if s is not None]
            ).to(device)
        )

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[not_done_mask] = self.target_net(not_done_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount) + reward_state_batch

        loss = self.DQN_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        logging.debug("loss: {}".format(loss.item()))

        # gradient clip
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        self.step += 1
        if self.step % self.lr_decay_every == 0 and self.step != 0:
            self.lr_scheduler.step()

        if update and self.step % self.update_target_every == 0 and self.step != 0:
            self.update_target_net()


        return loss.item()
