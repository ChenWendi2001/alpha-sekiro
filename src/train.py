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

from transition import State, Transition
from env import SekiroEnv

class Trainer():
    def __init__(self, config) -> None:

        if config.model_name == "":
            config.model_name = "{}-{}".format("test" if config.test_mode else "train", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.model_name = config.model_name
        self.ckpt_dir = os.path.join(config.model_dir, self.model_name)

    
        self.agent = Agent(config)
        self.config = config
        
        
        # epsilon
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_end = config.epsilon_end
        

        # prepare folder
        for dir in [config.log_dir, self.ckpt_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

        
        # tensorboard
        self.trainwriter = SummaryWriter(
            os.path.join(config.log_dir,
            self.model_name
            ),
            flush_secs=60
        )



    def run(self):
        '''the main training pipeline
        ''' 

        best_total_reward = 0
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
        damage_meter = AverageMeter("damage to boss")
        def stat_closure(damage):
            damage_meter.update(damage)
        
        for episode in trange(self.config.episodes):
            start_time = time.time()
            start_step = self.agent.step
            while True:

                # calculate latency
                logging.info('curr: {}, loop took {} seconds'.format(self.agent.step, time.time()-last_time))
                last_time = time.time()


                # remove useless frames like boss dying
                if emregency:
                    logging.critical("emergency happening!")
                else:
                    # Russian roulette
                    
                    if self.config.test_mode or random.random() >= self.epsilon:
                        random_action = False
                        pred = self.agent.act(cur_state)
                        logging.info(pred)
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
                    
                    start_action_time = time.time()
                    # exec action
                    env.step(action)
                    
                    # traing one step
                    if not self.config.test_mode and len(self.agent.replay_buffer) > self.config.start_train_after:
                        loss = self.agent.train_Q_network()
                        self.trainwriter.add_scalar("loss/train", loss, self.agent.step)
                    end_action_time = time.time()
                    if (end_action_time - start_action_time < 0.2):
                        time.sleep(0.2 - (end_action_time - start_action_time))

                # get next state
                next_obs, reward, done, emregency = env.obs(stat_closure)

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

            
            if not self.config.test_mode and ((episode + 1) % self.config.save_model_every == 0 or reward_meter.sum > best_total_reward):
             
                best_total_reward = max(reward_meter.sum, best_total_reward)
                torch.save(self.agent.policy_net.state_dict(), 
                    os.path.join(self.ckpt_dir, "{}-{}.pt".format(
                        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                        reward_meter.sum
                    ))
                )

            # tensorboard
            end_time = time.time()
            end_step = self.agent.step
            self.trainwriter.add_scalar("step_taken/train", end_step - start_step, episode)
            self.trainwriter.add_scalar("time_used/train", end_time - start_time, episode)
            self.trainwriter.add_scalar("damage_sum/train", damage_meter.sum, episode)
            self.trainwriter.add_scalar("reward_sum/train", reward_meter.sum, episode)
            self.trainwriter.add_scalar("reward_avg/train", reward_meter.avg, episode)

            # preset
            done = False
            last_time = time.time()
            reward_meter.reset()
            damage_meter.reset()

            obs = env.reset()
            cur_state = State(obs)





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

    for k in list(vars(config).keys()):
        logging.info('%s: %s' % (k, vars(config)[k]))
    
    trainer = Trainer(config)
    trainer.run()
