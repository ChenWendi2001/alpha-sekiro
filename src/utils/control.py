import win32api
import win32con
import time
import ctypes
import logging

from functools import partial

# virtual key codes map
# please refer to <https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes>
KEY_MAP = {}

# add alphabet
KEY_MAP.update({chr(c): 0x41+i for i, c in enumerate(range(ord('a'),ord('z')+1))})
KEY_MAP.update({chr(c): 0x41+i for i, c in enumerate(range(ord('A'),ord('Z')+1))})
KEY_MAP.update({"MID": 0x04}) # middle mouse button
KEY_MAP.update({"LEFT": 0x01}) # left mouse button
KEY_MAP.update({"SHIFT": 0x10})
KEY_MAP.update({"SPACE": 0x20})
print(KEY_MAP)

def press_key(num):
    MapVirtualKey = ctypes.windll.user32.MapVirtualKeyA
    time.sleep(0.05)
    win32api.keybd_event(num, MapVirtualKey(num, 0), 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(num, MapVirtualKey(num, 0), win32con.KEYEVENTF_KEYUP, 0)


attack = partial(press_key, KEY_MAP['J'])
defense = partial(press_key, KEY_MAP['k'])
dodge = partial(press_key, KEY_MAP['SHIFT'])
jump = partial(press_key, KEY_MAP['SPACE'])
lock = partial(press_key, KEY_MAP['MID'])
click = partial(press_key, KEY_MAP['LEFT'])

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
            logging.info('start game')
            time.sleep(1)
        else:
            paused = True
            logging.info('pause game')
            time.sleep(1)
    if paused:
        logging.info('paused press "T" in game to start')
        while True:
           
            # pauses game and can get annoying.
            if win32api.GetAsyncKeyState(ord('T')):
                if paused:
                    paused = False
                    logging.info('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

def take_action(action):
    '''press corresponding keys

    Args:
        action (int): action index

    '''
    [attack, defense, dodge, jump][action]()




if __name__ == "__main__":
    for i in range(10):
        press_key(KEY_MAP['Z'])