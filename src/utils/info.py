import numpy as np
import cv2
import time
from functools import partial

from config import *
from screenshot import fetch_image

def get_blood(image, height, width):
    image = fetch_image()
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blood_bar = image[height[0]:height[1], width[0]:width[1]]
    # cv2.imwrite("test.jpg", blood_bar)
    blood_bar = cv2.Canny(cv2.GaussianBlur(blood_bar,(3,3),0), 0, 100)

    # debug code
    # print(blood_bar.shape)
    # cv2.imwrite("test.jpg", blood_bar)
    # print(blood_bar.argmax(axis=-1))
    
    # FIXME: error when low blood 
    blood = np.median(blood_bar.argmax(axis=-1))
    return blood


get_self_blood = partial(get_blood, height=SELF_BLOOD_HEIGHT, width=SELF_BLOOD_WIDTH)

if __name__ == "__main__":
    last_time = time.time()
    while True:
        print(get_self_blood(fetch_image()))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()