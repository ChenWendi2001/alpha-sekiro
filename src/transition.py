from utils.info import *
from typing import Optional
from torchvision import transforms as T

class State():
    '''
        info about current state from observations
    '''
    def __init__(self, obs) -> None:
        '''generate state from obs(screen shot)

        Args:      
            observation(Tuple):
                focus_area      npt.NDArray[np.uint8], "L"
                agent_hp        float
                agent_ep        float
                boss_hp         float
        '''
        
        self.image = obs[0]
        self.agent_hp = obs[1]
        self.agent_ep = obs[3]
        self.boss_hp = obs[2]


class Transition():
    '''
        the minimal element in replay buffer
    '''
    def __init__(self, state: State, action: int, next_state: Optional[State], reward) -> None:
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