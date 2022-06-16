import logging
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import win32con
import win32gui

from .actions import Actor
from .env_config import AGENT_KEYMAP, GAME_NAME, REVIVE_DELAY
from .observation import Observer
from .utils import timeLog
from utils import control as Control

class SekiroEnv():
    def __init__(self) -> None:
        self.handle = win32gui.FindWindow(0, GAME_NAME)
        if self.handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()

        self.actor = Actor(self.handle)
        self.observer = Observer(self.handle)

        self.last_agent_hp = 0
        self.last_agent_ep = 0
        self.last_boss_hp = 0
        self.last_boss_ep = 0

    def actionSpace(self) -> List[int]:
        return list(range(len(AGENT_KEYMAP)))

    def __stepReward(self, obs: Tuple) -> float:
        agent_hp, boss_hp, agent_ep, boss_ep = obs[1:]

        if boss_hp < self.last_boss_hp:
            logging.info(f"Hurt Boss! {self.last_boss_hp: 1f} -> {boss_hp: 1f}")
        # TODO: refine reward
        rewards = np.array(
            [agent_hp - self.last_agent_hp,
             self.last_boss_hp - boss_hp,
             min(0, self.last_agent_ep - agent_ep),
             max(0, boss_ep - self.last_boss_ep)])
        weights = np.array([0.2, 0.2, 0.1, 0.1])
        reward = weights.dot(rewards).item()

        reward = -100 if agent_hp == 0 else reward

        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp
        self.last_boss_ep = boss_ep

        logging.info(f"reward: {reward:<.2f}")
        return reward

    @timeLog
    def step(self, action: int) -> Tuple[Tuple[npt.NDArray[np.uint8],
                                               float, float, float, float],
                                         float, bool, None]:
        """[summary]

        observation:
            focus_area      npt.NDArray[np.uint8], "L"
            agent_hp        float
            boss_hp         float
            agent_ep        float
            boss_ep         float

        Returns:
            observation           Tuple
            reward          float
            done            bool
            info            None
        """
        action_key = list(AGENT_KEYMAP.keys())[action]
        self.actor.agentAction(action_key)

        screen_shot = self.observer.shotScreen()
        obs = self.observer.getObs(screen_shot)

        done = obs[1] == 0
        if done:
            
            logging.info("player died!")
            time.sleep(10)
            self.actor.envAction("focus")
                
            self.actor.envAction("switch_full_blood")
            self.actor.envAction("switch_full_blood")
            self.actor.envAction("revive", action_delay=REVIVE_DELAY)
            # pause game
            self.actor.envAction("pause", action_delay=1)

        if obs[2] == 0:
            # TODO: succeed
            raise NotImplementedError()

        return obs, self.__stepReward(obs), done, None

    def reset(self) -> Tuple[npt.NDArray[np.uint8],
                             float, float, float, float]:
        # restore window
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND,
                             win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        # resume game 
        self.actor.envAction("resume", action_delay=True)

        
        # make agent full blood
        self.actor.envAction("switch_visible")
        self.actor.envAction("switch_invincible")
        self.actor.autoLock(self.observer.shotScreen, self.observer.getRawFocusArea)
        self.actor.envAction("switch_invincible")
        self.actor.envAction("switch_visible")
        

        screen_shot = self.observer.shotScreen()
        obs = self.observer.getObs(screen_shot)
        self.last_agent_hp, self.last_boss_hp, \
            self.last_agent_ep, self.last_boss_ep = obs[1:]

        return obs

