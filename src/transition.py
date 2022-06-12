

class State():
    '''
        info about current state from observations
    '''
    def __init__(self, obs):
        '''generate state from obs(screen shot)

        Args:
            obs (_type_): _description_
        '''
        raise NotImplementedError
    
    def toTensor(self):
        raise NotImplementedError


class Transition():
    '''
        the element in replay buffer
    '''
    def __init__(self, state: State, action, next_state: State, reward, done) -> None:
        '''_summary_

        Args:
            state (State): _description_
            action (_type_): _description_
            next_state (State): _description_
            reward (_type_): _description_
            done (function): _description_
        '''
        raise NotImplementedError