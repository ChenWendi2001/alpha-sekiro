import win32api
import win32con
import time
import ctypes
import logging

import pydirectinput

from functools import partial



keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_checks():
    keys = []
    for key in keyList:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def check_key_down(key):
    '''check whether a key is down

    Args:
        key (str): only one character
    '''
    if win32api.GetAsyncKeyState(ord(key)):
        return True
    else:
        return False



def wait_command(paused):
    if win32api.GetAsyncKeyState(ord('T')):
        if paused:
            paused = False
            logging.info('start dqn model')
            time.sleep(1)
        else:
            paused = True
            logging.info('pause dqn model')
            time.sleep(1)
    if paused:
        logging.info('paused press "T" in game to start dqn model')
        while True:
           
            # pauses game and can get annoying.
            if win32api.GetAsyncKeyState(ord('T')):
                if paused:
                    paused = False
                    logging.info('start dqn model')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused


if __name__ == "__main__":
    for i in range(10):
        pydirectinput.press('z')