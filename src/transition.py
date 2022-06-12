

class State():
    def __init__(self):
        raise NotImplementedError
        

class Transition():
    def __init__(self, state, action, next_state, reward, done) -> None:
        '''_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            next_state (_type_): _description_
            reward (_type_): _description_
            done (function): _description_
        '''
        raise NotImplementedError