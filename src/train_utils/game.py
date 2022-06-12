import random

import numpy as np
import torch
# from env.simulator import Simulator
from icecream import ic


def epsilonGreedy(q_values, epsilon=0.1):
    raise NotImplementedError
    ic.configureOutput(includeContext=True)
    printError(
        q_values.shape[-1] != len(indices),
        ic.format("shapes do not match!"))
    ic.configureOutput(includeContext=False)

    if np.random.rand() < epsilon:
        return random.choice(indices)
    else:
        return indices[q_values.argmax(axis=1).item()]


def playGame(net, epsilon, seed):
    # initialize seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_buffer = []
    env = Simulator()
    state = env.reset()

    while True:
        q_values = net.predict(state)
        action = epsilonGreedy(q_values, epsilon)
        next_state, reward, done, info = env.step(action)

        # collect data
        data_buffer.append(
            (state, reward, action, next_state, done))

        state = next_state
        # debug
        # env.drawBoard()

        if done:
            raise NotImplementedError
            env.score_board.showResults()
            return list(zip(*data_buffer))
