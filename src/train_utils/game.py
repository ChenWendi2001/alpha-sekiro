import random
from typing import List, Tuple

# from env.simulator import Simulator
import gym
import numpy as np
import torch


def epsilonGreedy(q_values, indices, epsilon=0.1):
    if np.random.rand() < epsilon:
        return random.choice(indices)
    else:
        return indices[q_values.argmax(axis=1).item()]


def playGame(net, epsilon, seed) -> Tuple[List, float]:
    # initialize seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_buffer = []
    env = gym.make("CartPole-v1")
    state = env.reset()
    score = 0

    while True:
        q_values = net.predict(state)
        action = epsilonGreedy(q_values, [0, 1], epsilon)
        next_state, reward, done, info = env.step(action)
        score += reward

        # collect data
        data_buffer.append(
            (state, reward, action, next_state, done))

        state = next_state
        # debug
        # env.render()

        if done:
            env.close()
            # env.score_board.showResults()
            return data_buffer, score
