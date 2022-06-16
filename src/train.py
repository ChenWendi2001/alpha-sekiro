import copy

import numpy as np
import torch
from tqdm import tqdm, trange
import random
import time
import datetime
import logging
from icecream import ic
import os

from agent import Agent
from config import Config
from utils import screenshot as Screenshot, control as Control
import reward as Reward
from transition import State, Transition
from env import SekiroEnv

class Trainer():
    def __init__(self, config) -> None:
        self.agent = Agent(config)
        self.config = config
        
        # epsilon
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_end = config.epsilon_end


    def run(self):
        '''the main training pipeline
        '''
        paused = True
        paused = Control.wait_command(paused)
        env = SekiroEnv()
        for episode in trange(self.config.episodes):
            # start a new game by pressing 'T' on game window


            # get first frame
            obs = env.reset()
            cur_state = State(obs)

            # preset
            done = False
            last_time = time.time()
            total_reward = 0


            while True:

                # calculate latency
                logging.info('curr: {}, loop took {} seconds'.format(self.agent.step, time.time()-last_time))
                last_time = time.time()

                # Russian roulette
                
                if random.random() >= self.epsilon:
                    random_action = False
                    pred = self.agent.act(cur_state)
                    action_sort = np.squeeze(np.argsort(pred)[::-1])
                    action = action_sort[0]
                else:
                    random_action = True
                    action = random.randint(0, config.action_dim - 1)


                logging.info("Action: {} [{}]".format(action, "random" if random_action else "learned"))


                # update epsilon
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay
                
                # get next state
                next_obs, reward, done, _ = env.step(action)
                next_state = State(obs=next_obs)

                self.agent.store_transition(Transition(
                    state=cur_state,
                    action=action,
                    next_state=(next_state if not done else None),
                    reward=reward
                ))

                cur_state = next_state


                # check "T" key 
                paused = Control.wait_command(paused)

                total_reward += reward
                if done:
                    break

                # traing one step
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    self.agent.train_Q_network()
            
            if episode % self.config.save_model_every == 0:
                if not os.path.exists(config.model_dir):
                    os.mkdir(config.model_dir)
                torch.save(self.agent.policy_net.state_dict(), os.path.join(config.model_dir, "{}.pt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))

if __name__ == "__main__":
    
    config = Config().parse()

    # logging setting
    ic.disable()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    trainer = Trainer(config)
    trainer.run()
