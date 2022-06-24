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

from tensorboardX import SummaryWriter

from agent import Agent
from config import Config
from utils import screenshot as Screenshot, control as Control
from utils.average_meter import AverageMeter
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

        # prepare folder
        for dir in [config.log_dir, config.model_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

        
        # tensorboard
        self.trainwriter = SummaryWriter(
            os.path.join(config.log_dir),
            "{}-{}".format("test" if self.config.test_mode else "train", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        )



    def run(self):
        '''the main training pipeline
        ''' 
        env = SekiroEnv()
        paused = True
        paused = Control.wait_command(paused)
        # Control.reset_cheater()
        # Control.infinite_respawn()

        # start a new game by pressing 'T' on game window
        # get first frame
        obs = env.reset()
        time.sleep(0.05)
        cur_state = State(obs)

        # preset
        done = False
        emregency = False
        last_time = time.time()
        reward_meter = AverageMeter("reward")
        reward_meter.reset()
        
        for episode in trange(self.config.episodes):
            
            while True:

                # calculate latency
                logging.info('curr: {}, loop took {} seconds'.format(self.agent.step, time.time()-last_time))
                last_time = time.time()


                # remove useless frames like boss dying
                if not emregency:
                    # Russian roulette
                    
                    if self.config.test_mode or random.random() >= self.epsilon:
                        random_action = False
                        pred = self.agent.act(cur_state)
                        action_sort = np.squeeze(np.argsort(pred)[::-1])
                        action = action_sort[0]
                    else:
                        random_action = True
                        random_value = random.random()
                        if random_value < 0.3:
                            action = 0
                        elif random_value < 0.5:
                            action = 1
                        elif random_value < 0.6:
                            action = 2
                        elif random_value < 0.7:
                            action = 3
                        elif random_value < 0.8:
                            action = 4
                        elif random_value < 0.9:
                            action = 5
                        else:
                            action = 6


                    logging.info("Action: {} [{}]".format(action, "random" if random_action else "learned"))


                    # update epsilon
                    if self.epsilon > self.epsilon_end:
                        self.epsilon *= self.epsilon_decay
                    
                    # exec action
                    env.step(action)
                    
                    # traing one step
                    if not self.config.test_mode and len(self.agent.replay_buffer) > self.config.batch_size:
                        loss = self.agent.train_Q_network()
                        self.trainwriter.add_scalar("loss/train", loss, episode)
                
                # get next state
                next_obs, reward, done, emregency = env.obs()

                next_state = State(obs=next_obs)


                if not self.config.test_mode: 
                    self.agent.store_transition(Transition(
                        state=cur_state,
                        action=action,
                        next_state=(next_state if not done else None),
                        reward=reward
                    ))

                cur_state = next_state


                # check "T" key 
                paused = Control.wait_command(paused)

                reward_meter.update(reward)
                
                if done:
                    break

            
            if (episode + 1) % self.config.save_model_every == 0:
                if not os.path.exists(config.model_dir):
                    os.mkdir(config.model_dir)
                torch.save(self.agent.policy_net.state_dict(), os.path.join(config.model_dir, "{}.pt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))

            obs = env.reset()
            cur_state = State(obs)

            # preset
            done = False
            last_time = time.time()
            reward_meter.reset()

            # tensorboard
            self.trainwriter.add_scalar("reward_sum/train", reward_meter.sum, episode)
            self.trainwriter.add_scalar("reward_avg/train", reward_meter.avg, episode)

if __name__ == "__main__":
    
    config = Config().parse()

    # logging setting
    ic.disable()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(config.log_dir, "{}.txt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    trainer = Trainer(config)
    trainer.run()
