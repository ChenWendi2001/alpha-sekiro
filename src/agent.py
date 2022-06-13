
import torch
from collections import deque
import random
import logging
import numpy as np
from typing import List


from transition import Transition, State
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory():
    '''replay memory with fixed capacity

    '''
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

    def sample(self, batch_size: int) -> List[Transition]:
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
        self.discount = config.discount

        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.lr_decay_every = config.lr_decay_every
        self.update_target_every = config.update_target_every



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
    
    def act(self, state: State):
        '''act on input state 

        Args:
            state (State): _description_

        Returns:
            numpy: predicted prob on input state
        '''
        state = (
            torch.from_numpy(state.image).float().to(device).unsqueeze(0),
        )

        self.policy_net.eval()
        with torch.no_grad():
            out = self.policy_net(state)
            out = out.cpu().squeeze(0).numpy()
        self.policy_net.train()
        
        return out

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
        
        state_batch = (
            torch.tensor(
                [state.image for state in state_batch]
            ),
        )

        # action batch
        action_batch = torch.tensor(
            [data.action for data in minibatch], device=device
        ).long()

        # next state batch
        next_state_batch = [data.next_state for data in minibatch]

        # reward batch
        reward_state_batch = torch.tensor(
            [data.reward for data in minibatch], device=device
        ).float()

        reward_state_batch = reward_state_batch.repeat_interleave(
            self.state_dim, 0
        ).view(self.batch_size, self.state_w, self.state_h)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        not_done_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state_batch)), 
            device=device,
            dtype=torch.bool
        )

        not_done_next_states = torch.cat([
            s for s in next_state_batch if s is not None
        ])

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[not_done_mask] = self.target_net(not_done_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount) + reward_state_batch

        loss = self.DQN_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # gradient clip
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        self.step += 1
        if self.step % self.lr_decay_every == 0 and self.step != 0:
            self.lr_scheduler.step()

        if update and self.step % self.update_target_every == 0 and self.step != 0:
            self.update_target_net()


        return loss.item()


