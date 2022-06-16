import logging
import time

import pydirectinput

from .utils import timeLog

from .env_config import AGENT_KEYMAP, ENV_KEYMAP


class Actor():
    def __init__(self, handle) -> None:
        self.handle = handle
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    @timeLog
    def agentAction(self, key,
                    action_delay: float = 0):
        if key not in self.agent_keymap:
            logging.critical("invalid agent action")
            raise RuntimeError()

        pydirectinput.press(self.agent_keymap[key])
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
