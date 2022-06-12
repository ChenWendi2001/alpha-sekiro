
import torch
from collections import deque
import random
import logging
import numpy as np


from transition import Transition
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        '''_summary_

        Args:
            capacity (int): capacity of replay memory
        '''
        self.memory = deque([], maxlen=capacity)

    def store(self, transition: Transition):
        '''_summary_

        Args:
            transition (Transition): transition to be added
        '''
        self.memory.append(transition)

    def sample(self, batch_size: int):
        '''_summary_

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
        self.state_dim = config.obs_height * config.obs_width
        self.state_w = config.obs_width
        self.state_h = config.obs_height

        self.action_dim = config.action_dim

        self.model_dir = config.model_dir

        self.batch_size = config.batch_size

        # Replay Buffer
        self.replay_buffer = ReplayMemory(capacity=config.capacity)
        # Policy Net
        self.policy_net = Model(config).to(device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
        self.DQN_criterion = F.mse_loss
        # Target Net
        self.target_net = Model(config).to(device)

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
    
    def trian_Q_network(self, update=True):
        '''_summary_

        Args:
            update (bool, optional): whether to update target model. Defaults to True.
        '''

        minibatch = self.replay_buffer.sample(self.batch_size)

        np.random.shuffle(minibatch)
        minibatch = minibatch.squeeze(-1)
        # state batch
        state_batch = [data.state for data in minibatch]
        
        raise NotImplementedError