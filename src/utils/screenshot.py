import win32api
import win32con
import win32gui

from PIL import Image, ImageGrab
import numpy as np

def get_window_pos(name):
    handle = win32gui.FindWindow(0, name)


    if handle == 0:
        return None
    else:
        import ctypes
        from ctypes.wintypes import HWND, DWORD, RECT
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware(2)
        dwmapi = ctypes.WinDLL("dwmapi")
        DMWA_EXTENDED_FRAME_BOUNDS = 9
        rect = RECT()
        dwmapi.DwmGetWindowAttribute(HWND(handle), DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
                                ctypes.byref(rect), ctypes.sizeof(rect))
        curr_anchor = win32gui.GetWindowRect(handle)
        return (rect.left, rect.top, rect.right, rect.bottom), handle

def fetch_image():
    anchor, handle = get_window_pos('Sekiro')
    # restore window
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # focus
    win32gui.SetForegroundWindow(handle)
    grab_image = ImageGrab.grab(anchor)
    grab_image = np.array(grab_image)
    
    grab_image = grab_image[-721:-1,1:-1,::-1]
    # print(grab_image.size)
    return grab_image

if __name__ == "__main__":
    fetch_image()