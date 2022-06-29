import logging
import time
from xmlrpc.client import Boolean

from typing import Tuple

import pydirectinput

from .utils import timeLog
from .memory import Memory

from .env_config import AGENT_KEYMAP, ENV_KEYMAP

import cv2
import numpy as np

class Actor():
    def __init__(self, handle, memory) -> None:
        self.handle = handle
        self.memory: Memory = memory
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    @timeLog
    def agentAction(self, key,
                    action_delay: float = 0):
        if key not in self.agent_keymap:
            logging.critical("invalid agent action")
            raise RuntimeError()

        if isinstance(self.agent_keymap[key], str):
            pydirectinput.press(self.agent_keymap[key])

        if isinstance(self.agent_keymap[key], tuple):
            (hold, inst) = self.agent_keymap[key]
            pydirectinput.keyDown(hold)
            pydirectinput.press(inst)
            pydirectinput.keyUp(hold)

        
        logging.info(f"action: {key}")

        time.sleep(action_delay)

    @timeLog
    def envAction(self, key,
                  action_delay: float = 0):
        if key not in self.env_keymap:
            logging.critical("invalid env action")
            raise RuntimeError()

        pydirectinput.press(self.env_keymap[key])
        logging.debug(f"env: {key}")

        time.sleep(action_delay)

    def autoLock(self):
        
        def adjustHorizon(i) -> None:
            pydirectinput.keyDown('o')
            time.sleep(0.25 * (i+1))
            pydirectinput.keyUp('o')


        locked = self.memory.lockBoss()
        while not locked:
            i = 0
            while not locked:
                adjustHorizon(i)
                locked = self.memory.lockBoss()
                i += 1
            