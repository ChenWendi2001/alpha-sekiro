import copy

import numpy as np
import torch
from icecream import ic
from tqdm import tqdm

from agent.d3qn import D3QN
from train_utils.game import playGame
from train_utils.replay_buffer import ReplayBuffer


class Trainer():
    def __init__(self) -> None:
        self.net = D3QN()
        self.cnt = 0  # version of best net
        self.best_net = copy.deepcopy(self.net)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = 0.01

    def collectData(self):
        for _ in tqdm(range(2)):
            episode_data = playGame(
                self.net, self.epsilon,
                np.random.randint(2 ** 30))
            self.replay_buffer.add(episode_data)

        # self.epsilon -= TRAIN_CONFIG.delta_epsilon
        # self.epsilon = max(self.epsilon, TRAIN_CONFIG.min_epsilon)

    def train(self):
        ic("train model")

        total_num, mean_loss = 0, 0
        epochs = 2
        for _ in tqdm(range(epochs)):
            data_batch = self.replay_buffer.sample()
            loss = self.net.trainStep(data_batch)
            total_num += data_batch[-1].shape[0]
            mean_loss += loss * data_batch[-1].shape[0]
        mean_loss /= total_num
        ic(mean_loss)

    def run(self):
        """[summary]
        pipeline: collect data, train, evaluate, update and repeat
        """
        for i in range(1, 1000 + 1):
            # >>>>> collect data
            self.collectData()
            print("Round {} finish, buffer size {}".format(
                i, self.replay_buffer.size()))
            # save data
            if i % 100 == 0:
                self.replay_buffer.save(version=i)

            # >>>>> train
            if self.replay_buffer.enough():
                self.train()

            # >>>>> evaluate
            if i % 20 == 0:
                raise NotImplementedError
                win_rate = self.evaluate()
                if win_rate >= TRAIN_CONFIG.update_threshold:
                    self.cnt += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=self.cnt)
                    message = "new best model {}!".format(self.cnt)
                    ic(message)
                else:
                    ic("reject new model.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
