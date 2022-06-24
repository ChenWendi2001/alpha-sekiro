import logging
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import win32con
import win32gui

from .actions import Actor
from .env_config import AGENT_KEYMAP, GAME_NAME, REVIVE_DELAY, AGENT_DEAD_DELAY, ROTATION_DELAY, MAP_CENTER
from .observation import Observer
from .utils import timeLog
from utils import control as Control
from .memory import Memory

class SekiroEnv():
    def __init__(self) -> None:
        self.handle = win32gui.FindWindow(0, GAME_NAME)
        if self.handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()


        self.memory = Memory()
        self.actor = Actor(self.handle, self.memory)
        self.observer = Observer(self.handle, self.memory)

        self.last_agent_hp = 0
        self.last_agent_ep = 0
        self.last_boss_hp = 0

        self.boss_dead = False

    def actionSpace(self) -> List[int]:
        return list(range(len(AGENT_KEYMAP)))

    def __stepReward(self, obs: Tuple) -> float:
        agent_hp, agent_ep, boss_hp, _ = obs[1:]

        if boss_hp < self.last_boss_hp:
            logging.info(f"Hurt Boss! {self.last_boss_hp: 1f} -> {boss_hp: 1f}")
        # TODO: refine reward
        rewards = np.array(
            [agent_hp - self.last_agent_hp,
             self.last_boss_hp - boss_hp,
             min(0, self.last_agent_ep - agent_ep)])
        weights = np.array([200, 1000, 50])
        reward = weights.dot(rewards).item()

        reward = -50 if agent_hp == 0 else reward

        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp

        logging.info(f"reward: {reward:<.2f}")
        return reward

    def step(self, action: int): 
        action_key = list(AGENT_KEYMAP.keys())[action]
        self.actor.agentAction(action_key)

    @timeLog
    def obs(self) -> Tuple[Tuple[npt.NDArray[np.uint8],
                                               float, float, float],
                                         float, bool, None]:
        """[summary]

        observation:
            focus_area      npt.NDArray[np.uint8], "L"
            agent_hp        float
            agent_ep        float
            boss_hp         float

        Returns:
            observation           Tuple
            reward          float
            done            bool
            info            None
        """
        # checking lock
        lock_state = self.memory.lockBoss()
        logging.info(f"lock state: {lock_state}")


        screen_shot = self.observer.shotScreen()
        obs = self.observer.getObs(screen_shot)
        reward = self.__stepReward(obs)
        done = obs[1] == 0
        if done:
            
            logging.info("player died!")
            time.sleep(AGENT_DEAD_DELAY)
            self.memory.lockBoss()
            time.sleep(ROTATION_DELAY)
            self.memory.reviveAgent(need_delay=True)
            # pause game
            self.actor.envAction("pause", action_delay=0.5)

        if obs[3] < 0.05:
            logging.info("Boss died! The dqn will be paused!")
            self.memory.reviveBoss()
            logging.info("Boss revived! The dqn will be resumed")
            self.last_boss_hp = 1.0

        return obs, reward, done, False

    def reset(self) -> Tuple[npt.NDArray[np.uint8],
                             float, float, float]:
        # restore window
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND,
                             win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        # auto focus

        # resume game 
        self.actor.envAction("resume", action_delay=0.5)

        self.memory.transportAgent(MAP_CENTER)
        self.actor.autoLock()

        self.memory.reviveAgent(need_delay=False)
        

        screen_shot = self.observer.shotScreen()
        obs = self.observer.getObs(screen_shot)
        self.last_agent_hp, self.last_boss_hp, \
            self.last_agent_ep, _ = obs[1:]

        return obs

