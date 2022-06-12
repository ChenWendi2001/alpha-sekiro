import copy

from tensorboardX import SummaryWriter

import numpy as np
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
        self.epsilon = 0.5

        import datetime
        log_name = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(f"./results/{log_name}")

    def collectData(self, epoch):
        n_game, mean_score = 1, 0
        for _ in tqdm(range(n_game)):
            episode_data, score = playGame(
                self.net, self.epsilon,
                np.random.randint(2 ** 30))
            self.replay_buffer.add(episode_data)
            mean_score += score
        mean_score /= n_game
        self.writer.add_scalar("score", mean_score, epoch)

        self.epsilon -= 0.01
        self.epsilon = max(self.epsilon, 0.01)

    def train(self, epoch):
        epochs = 10
        total_num, mean_loss = 0, 0
        for _ in tqdm(range(epochs)):
            data_batch = self.replay_buffer.sample()
            loss = self.net.trainStep(data_batch)
            total_num += data_batch[-1].shape[0]
            mean_loss += loss * data_batch[-1].shape[0]
        mean_loss /= total_num
        self.writer.add_scalar("loss", mean_loss, epoch)

    def run(self):
        """[summary]
        pipeline: collect data, train, evaluate, update and repeat
        """
        for i in range(1, 200 + 1):
            # >>>>> collect data
            self.collectData(i)
            print("Round {} finish, buffer size {}".format(
                i, self.replay_buffer.size()))
            # save data
            if i % 100 == 0:
                self.replay_buffer.save(version=i)

            # >>>>> train
            if self.replay_buffer.enough():
                self.train(i)

            # >>>>> evaluate
            if i % 20 == 0:
                # self.cnt += 1
                self.best_net = copy.deepcopy(self.net)
                self.best_net.save(version=self.cnt)
                continue
                win_rate = self.evaluate()
                if win_rate >= TRAIN_CONFIG.update_threshold:
                    self.cnt += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=self.cnt)
                    message = "new best model {}!".format(self.cnt)
                    ic(message)
                else:
                    ic("reject new model.")

            self.writer.flush()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
