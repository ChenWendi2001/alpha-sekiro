import numpy as np
import cv2
import time
from functools import partial

from const import *
from screenshot import fetch_image

def get_blood(image, height, width):
    image = fetch_image()
    # cv2.imwrite("test.jpg", image)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blood_bar = image[height[0]:height[1], width[0]:width[1]]
    # cv2.imwrite("test.jpg", blood_bar)
    blood_bar = cv2.Canny(cv2.GaussianBlur(blood_bar,(5,5),0), 0, 100)

    # debug code
    # print(blood_bar.shape)
    # cv2.imwrite("test.jpg", blood_bar)
    # print(blood_bar.argmax(axis=-1))
    
    # FIXME: error when low blood 
    blood = np.median(blood_bar.argmax(axis=-1))
    return blood

def get_endurance(image, height, width):
    THRESHOLD = 171

    image = fetch_image()
    # cv2.imwrite("test.jpg", image)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image_gray[(height[0] + height[1])//2, width[0]-5] < THRESHOLD:
        return 0
    
    endurance_bar = image[height[0]:height[1], width[0]:width[1]]
    # cv2.imwrite("test.jpg", endurance_bar)
    endurance_bar = cv2.Canny(cv2.GaussianBlur(endurance_bar,(3,3),0), 0, 100)

    # debug code
    # print(endurance_bar.shape)
    # cv2.imwrite("test.jpg", endurance_bar)
    # print(endurance_bar.argmax(axis=-1))
     
    endurance = np.median(endurance_bar.argmax(axis=-1))
    return endurance

def get_state(image):
    return image[STATE_HEIGHT[0]:STATE_HEIGHT[1], STATE_WIDTH[0]:STATE_WIDTH[1]]


get_self_blood = partial(get_blood, height=SELF_BLOOD_HEIGHT, width=SELF_BLOOD_WIDTH)
get_self_endurance = partial(get_endurance, height=SELF_ENDURANCE_HEIGHT, width=SELF_ENDURANCE_WIDTH)

get_boss_blood = partial(get_blood, height=BOSS_BLOOD_HEIGHT, width=BOSS_BLOOD_WIDTH)
get_boss_endurance = partial(get_endurance, height=BOSS_ENDURANCE_HEIGHT, width=BOSS_ENDURANCE_WIDTH)

if __name__ == "__main__":
    last_time = time.time()
    while True:
        image = fetch_image()
        print("self blood:", get_self_blood(image))
        print("self endurance:", get_self_endurance(image))

        print("boss blood:", get_boss_blood(image))
        print("boss endurance:", get_boss_endurance(image))

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()