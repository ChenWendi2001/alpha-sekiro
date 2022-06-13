from transition import State
def get_reward(old_state: State, new_state: State, self_blood_animation_state: int):
    '''_summary_

    Args:
        old_state (State): state before specific action
        new_state (State): state after specific action
        self_blood_animation_state (int): 0: not in animation; 1: in animation

    Return:
        reward (int)
        done (bool) 
        self_blood_animation_state (int): new state
    '''
    if new_state.self_blood < 3: # self dead
        reward = -10
        done = True
    elif new_state.boss_blood - old_state.boss_blood > 15: # boss dead
        reward = 20
        done = False
    else:
        self_blood_reward = 0
        boss_blood_reward = 0

        if new_state.self_blood - old_state.self_blood < -7:
            if self_blood_animation_state == 0:
                self_blood_reward = -6
                self_blood_animation_state = 1
        else:
            self_blood_animation_state = 0
        
        if new_state.boss_blood - old_state.boss_blood <= -3:
            boss_blood_reward = 4
        
        reward = self_blood_reward + boss_blood_reward
        done = False

    return reward, done, self_blood_animation_state