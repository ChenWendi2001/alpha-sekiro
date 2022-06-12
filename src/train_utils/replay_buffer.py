import pickle
from collections import deque
from typing import List

import numpy as np
import torch


class ReplayBuffer():
    """[summary]
    states, rewards, actions, next_states, dones
    """

    def __init__(self) -> None:
        self.buffer = deque(maxlen=1000000)

    def size(self):
        return len(self.buffer)

    def save(self, version):
        dataset_dir = "dataset/"

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        print("save replay buffer version({})".format(version))
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_dir):
        print("load replay buffer {}".format(data_dir))
        with open(data_dir, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        """[summary]
        whether data is enough to start training
        """
        return self.size() > 100

    def add(self, episode_data: List):
        self.buffer.extend(episode_data)

    def sample(self):
        batch_size = 64
        indices = np.random.choice(
            len(self.buffer), batch_size)
        data_batch = map(
            lambda x: torch.from_numpy(np.stack(x)),
            zip(*[self.buffer[i] for i in indices])
        )
        return list(data_batch)
