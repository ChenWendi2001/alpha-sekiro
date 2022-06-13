from utils.info import *
from typing import int, Optional

class State():
    '''
        info about current state from observations
    '''
    def __init__(self, obs) -> None:
        '''generate state from obs(screen shot)

        Args:
            obs (numpy array): 
        '''
        self.image = obs
        self.self_blood = get_blood(obs)
        self.boss_blood = get_boss_blood(obs)
        self.self_endurance = get_self_endurance(obs)
        self.boss_endurance = get_boss_endurance(obs)


class Transition():
    '''
        the minimal element in replay buffer
    '''
    def __init__(self, state: State, action: int, next_state: State, reward) -> None:
        '''

        Args:
            state (State): _description_
            action (int): _description_
            next_state (State | None): _description_
            reward (int): _description_
        '''
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward