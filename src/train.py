import copy

import numpy as np
import torch
from tqdm import tqdm, trange
import random
import time
import datetime
import logging
import os

from agent import Agent
from config import Config
from utils import screenshot as Screenshot, control as Control
import reward as Reward
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
        '''the main training pipeline
        '''
        paused = True
        paused = Control.wait_command(paused)
        Control.lock()
        for episode in trange(self.config.episodes):
            # start a new game by pressing 'T' on game window
            time.sleep(2)


            # get first frame
            obs = Screenshot.fetch_image()
            cur_state = State(obs)

            # preset
            done = False
            last_time = time.time()
            total_reward = 0

            # avoid multi-judging during the animation of blood decreaing 
            # 0: not in animation, 1: in animation
            self_blood_animation_state = 0

            while True:
                # calculate latency
                logging.info('loop took {} seconds'.format(time.time()-last_time))
                last_time = time.time()

                # Russian roulette
                if random.random() >= self.epsilon:
                    pred = self.agent.act(cur_state)
                    action_sort = np.squeeze(np.argsort(pred)[::-1])
                    action = action_sort[0]
                else:
                    action = random.randint(0, config.action_dim - 1)


                logging.info("Action: {}".format(action))
                Control.take_action(action)

                # update epsilon
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay
                
                # get next state
                next_obs = Screenshot.fetch_image()
                next_state = State(obs=next_obs)

                # calculate reward and judge result
                reward, done, self_blood_animation_state = Reward.get_reward(
                    cur_state, 
                    next_state,
                    self_blood_animation_state
                    )

                if done:
                    logging.info("player died!")
                else:
                    logging.info("reward: {}".format(reward))

                self.agent.store_transition(Transition(
                    state=cur_state,
                    action=action,
                    next_state=(next_state if not done else None),
                    reward=reward
                ))

                # traing one step
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    self.agent.train_Q_network()

                # check "T" key 
                paused = Control.wait_command(paused)

                total_reward += reward
                if done == 1:

                    time.sleep(7)
                    Control.lock()
                    time.sleep(0.5)
                    Control.click()
                    break
            
            if episode % self.config.save_model_every:
                if not os.path.exists(config.model_dir):
                    os.mkdir(config.model_dir)
                torch.save(self.agent.policy_net.state_dict(), os.path.join(config.model_dir, "{}.pt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))

if __name__ == "__main__":
    
    config = Config().parse()

    # logging setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    trainer = Trainer(config)
    trainer.run()
