import win32api
import win32con
import time
import ctypes

from functools import partial

# virtual key codes map
# please refer to <https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes>
KEY_MAP = {}

# add alphabet
KEY_MAP.update({chr(c): 0x41+i for i, c in enumerate(range(ord('a'),ord('z')+1))})
KEY_MAP.update({chr(c): 0x41+i for i, c in enumerate(range(ord('A'),ord('Z')+1))})
print(KEY_MAP)

def press_key(num):
    MapVirtualKey = ctypes.windll.user32.MapVirtualKeyA
    time.sleep(0.05)
    win32api.keybd_event(num, MapVirtualKey(num, 0), 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(num, MapVirtualKey(num, 0), win32con.KEYEVENTF_KEYUP, 0)


attack = partial(press_key, KEY_MAP['J'])
defense = partial(press_key, KEY_MAP['k'])
lock = partial(press_key, KEY_MAP['L'])
dodge = partial(press_key, KEY_MAP['N'])
jump = partial(press_key, KEY_MAP['M'])

if __name__ == "__main__":
    for i in range(10):
        press_key(KEY_MAP['Z'])