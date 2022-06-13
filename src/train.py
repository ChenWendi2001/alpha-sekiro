import copy

import numpy as np
import torch
from tqdm import tqdm, trange
import random
import time
import datetime

from agent import Agent
from config import Config
from utils import screenshot as Screenshot, control as Control
from utils import reward as Reward
from transition import State, Transition


class Trainer():
    def __init__(self, config) -> None:
        self.agent = Agent(config)
        self.config = config
        
        # epsilon
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_end = config.epsilon_end


    def run(self):
        '''_summary_
        '''
        paused = True
        paused = Control.pause_game(paused)
       
        for episode in trange(self.config.episodes):
            # first frame
            obs = Screenshot.fetch_image()
            cur_state = State(obs=obs)
            done = False

            last_time = time.time()
            total_reward = 0
            while True:
                last_time = time.time()
                if random.random() >= self.epsilon:
                    pred = self.agent.act(cur_state)
                    action_sort = np.squeeze(np.argsort(pred)[::-1])
                    action = action_sort[0]

                else:
                    action = random.randint(0, config.action_dim)

                Control.take_action(action)
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay
                next_obs = Screenshot.fetch_image()
                next_state = State(obs=next_obs)
                reward, done = Reward.get_reward(cur_state, next_state)
                self.agent.store_transition(Transition(
                    state=cur_state,
                    action=action,
                    next_state=next_state,
                    reward=reward
                ))
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    self.agent.trian_Q_network()
                paused = Control.pause_game(paused)
                if done == 1:
                    break
            if episode % self.config.save_model_every:
                torch.save(self.agent.policy_net.state_dict(), "{}.pt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

if __name__ == "__main__":
    config = Config().parse()
    trainer = Trainer(config)
    trainer.run()
