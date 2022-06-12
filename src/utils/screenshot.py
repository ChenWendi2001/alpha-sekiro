import win32api
import win32con
import win32gui

from PIL import Image, ImageGrab

def get_window_pos(name):
    handle = win32gui.FindWindow(0, name)

    if handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(handle), handle

def fetch_image():
    anchor, handle = get_window_pos('Sekiro')
    # restore window
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # focus
    win32gui.SetForegroundWindow(handle)
    grab_image = ImageGrab.grab(anchor)
    # grab_image.save("test.jpg")
    # print(grab_image.size)
    return grab_image

if __name__ == "__main__":
    fetch_image()