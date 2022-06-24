import ctypes
import logging
import pickle
from datetime import datetime
from typing import Tuple
import os
import cv2

import numpy as np
import numpy.typing as npt
from icecream import ic
from PIL import Image, ImageGrab

from mmpose.apis import (init_pose_model, inference_top_down_pose_model, vis_pose_result)
from mmdet.apis import init_detector, inference_detector

from .utils import timeLog
from .memory import Memory
from .env_config import (AGENT_EP_ANCHOR, AGENT_HP_ANCHOR, BOSS_EP_ANCHOR,
                         BOSS_HP_ANCHOR, FOCUS_ANCHOR, FOCUS_SIZE,
                         SCREEN_ANCHOR, SCREEN_SIZE,
                         SELF_BLOOD_WIDTH, SELF_BLOOD_HEIGHT,
                         SELF_ENDURANCE_HEIGHT, SELF_ENDURANCE_WIDTH,
                         BOSS_BLOOD_HEIGHT, BOSS_BLOOD_WIDTH,
                         BOSS_ENDURANCE_HEIGHT,BOSS_ENDURANCE_WIDTH)




def get_blood(image:np.array, height:Tuple, width: Tuple):
    blood_bar = image[height[0]:height[1], width[0]:width[1]]
    blood_bar = cv2.cvtColor(blood_bar, cv2.COLOR_RGB2BGR)
    blood_edge = cv2.Canny(cv2.GaussianBlur(blood_bar,(5,5),0), 0, 100)

    # FIXME: error when low blood √
    # NOTE: check that the blood should be red
    blood = int(np.median(blood_edge.argmax(axis=-1)))
    sample = blood_bar[:, blood // 2]
    
    blood = int(blood / (width[1] - width[0]) * 100)
    red = np.array([[46, 61, 124]])
    return blood if np.linalg.norm(red - sample, 2) < 200 else 0

class Observer():
    """[summary]
    yield raw observation
    """

    def __init__(self, handle, memory: Memory) -> None:
        self.handle: int = handle
        self.memory: Memory = memory

        anchor = ctypes.wintypes.RECT()
        ctypes.windll.user32.SetProcessDPIAware(2)
        DMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(self.handle),
            ctypes.wintypes.DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(anchor), ctypes.sizeof(anchor))
        self.anchor = (anchor.left, anchor.top, anchor.right, anchor.bottom)
        logging.debug(anchor)

        self.timestamp: str = ""

        # HACK: load preset hp & ep
        self.asset_path = os.path.join(os.path.dirname(__file__), "asset")
        self.debug_path = os.path.join(os.path.dirname(__file__), "debug")
        if ic.enabled and not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        '''
        self.agent_hp_full = pickle.load(
            open(os.path.join(self.asset_path, "agent-hp-full.pkl"), "rb"))
        self.boss_hp_full = pickle.load(
            open(os.path.join(self.asset_path, "boss-hp-full.pkl"), "rb"))
        self.agent_ep_full = pickle.load(
            open(os.path.join(self.asset_path, "agent-ep-full.pkl"), "rb"))
        self.boss_ep_full = pickle.load(
            open(os.path.join(self.asset_path,"boss-ep-full.pkl"), "rb"))
        '''

        # load pose model
        root_path = os.path.join(os.path.dirname(__file__), "..", "..", "pretrained_model")
        detect_config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        detect_checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.detect_model = init_detector(os.path.join(root_path, detect_config_file), \
                os.path.join(root_path, detect_checkpoint_file), device='cuda:0')

        pose_config_file = 'topdown_heatmap_vipnas_mbv3_coco_256x192.py'
        pose_checkpoint_file = 'vipnas_mbv3_coco_256x192-7018731a_20211122.pth'
        self.pose_model = init_pose_model(os.path.join(root_path, pose_config_file), \
            os.path.join(root_path, pose_checkpoint_file), device='cuda:0')
        logging.info("Successfully loaded pose model!")

    def __select(self, arr: npt.NDArray, anchor: Tuple) -> npt.NDArray:
        # NOTE: C x H x W
        left, top, right, bottom = anchor
        return arr[:, top:bottom, left:right]

    # @timeLog
    def shotScreen(self) -> npt.NDArray[np.int16]:
        screen_shot = ImageGrab.grab(self.anchor)
        # NOTE: C x H x W, "RGB"
        screen_shot = np.array(screen_shot, dtype=np.int16).transpose(2, 0, 1)
        screen_shot = self.__select(screen_shot, SCREEN_ANCHOR)

        if ic.enabled:
            self.timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            Image.fromarray(
                screen_shot.transpose(1, 2, 0).astype(np.uint8)).save(
                    os.path.join(self.debug_path,f"/screen-shot-{self.timestamp}.png"))


        if screen_shot.shape[1:] != SCREEN_SIZE:
            logging.critical("incorrect screenshot")
            raise RuntimeError()

        return screen_shot

    def __calcProperty(self, arr: npt.NDArray[np.int16],
                       target: npt.NDArray[np.int16], threshold, prefix="") -> float:
        """[summary]

        Args:
            arr (npt.NDArray[np.int16]): C x H x W
            target (npt.NDArray[np.int16]): C x H x W
        """
        if ic.enabled:
            Image.fromarray(
                arr.transpose(1, 2, 0).astype(np.uint8), mode="HSV").convert(
                    "RGB").save(os.path.join(self.debug_path, f"{prefix}-{self.timestamp}.png"))
        if arr.shape != target.shape:
            logging.critical("incorrect arr shape")
            raise RuntimeError()

        result: npt.NDArray[np.bool_] = np.max(
            np.abs(target - arr), axis=0) < (threshold * 256)
        if ic.enabled:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1)
            ax[0].spy(result)
            ax[1].imshow(Image.fromarray(
                arr.transpose(1, 2, 0).astype(np.uint8), mode="HSV").convert("RGB"))
            fig.subplots_adjust(hspace=-0.8)
            plt.savefig(os.path.join(self.debug_path, f"{prefix}-content-{self.timestamp}.png"))
            plt.close()
        result = np.sum(result, axis=0) > result.shape[0] / 2

        return 100 * np.sum(result) / result.size

    @timeLog
    def getObs(self, screen_shot: npt.NDArray[np.int16]) -> \
            Tuple[npt.NDArray[np.uint8], float, float, float, float]:
        """[summary]

        Observation:
            image           npt.NDArray[np.uint8]
            agent_hp        float
            agent_ep        float
            boss_hp         float
        """
        # NOTE: use HSV
        hsv_screen_shot = np.array(Image.fromarray(
            screen_shot.astype(np.uint8).transpose(1, 2, 0)).convert("HSV"),
            dtype=np.int16).transpose(2, 0, 1)

        agent_hp, agent_ep, boss_hp = self.memory.getStatus()

        logging.info(f"agent hp: {agent_hp:.1f}, boss hp: {boss_hp:.1f}")
        logging.info(f"agent ep: {agent_ep:.1f}")

        focus_area = Image.fromarray(self.__select(
            screen_shot, FOCUS_ANCHOR).transpose(1, 2, 0).astype(np.uint8))
        if ic.enabled:
            focus_area.save(os.path.join(self.debug_path, f"focus-{self.timestamp}.png"))
        focus_area = np.array(
            focus_area.resize(FOCUS_SIZE), dtype=np.uint8).transpose(2, 0, 1)
        
        # detection and pose
        input_focus_area = focus_area.transpose(1, 2, 0)[:,:,::-1]
        detect_result = inference_detector(self.detect_model, input_focus_area)
        bbox = detect_result[0][:1]
        bbox = [{'bbox': bb} for bb in bbox]

        pose_result, _ = inference_top_down_pose_model(self.pose_model, input_focus_area, bbox, format="xyxy")

        # print("pose!", pose_result)
        if ic.enabled:
            vis_pose_result(self.pose_model, input_focus_area, pose_result, \
                 out_file=os.path.join(self.debug_path,f"pose-{self.timestamp}.png"))

        return focus_area, agent_hp, agent_ep, boss_hp

    def getRawFocusArea(self, screen_shot: npt.NDArray[np.int16]) -> \
            npt.NDArray[np.uint8]:
        
        return self.__select(
            screen_shot, FOCUS_ANCHOR).astype(np.uint8).transpose(1, 2, 0)

