import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from typing import Callable, List, Tuple


def load_bin_image(path: str, inv=False) -> np.ndarray:
    """
    Loads image and turn it into an image of 0s and 1s
    """
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray <= 10] = inv
    img_gray[img_gray > 10] = 1 - inv
    img_gray = img_gray.astype(np.int8)
    return img_gray


def mcircle_mask(image_shape: Tuple[int, int]) -> Callable[[int, int, int], np.ndarray]:
    """
    Defines variables used inside circle_mask
    """
    x = np.arange(image_shape[0]).reshape(1,-1)
    y = np.arange(image_shape[1]).reshape(-1,1)
    img_shape = image_shape

    @lru_cache(maxsize=1024)
    def circle_mask(cx: int, cy: int, r: int) -> np.ndarray:
        """
        Creates a circular mask at the given center
        coordinates
        """
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        circle = np.zeros(img_shape, dtype=np.int8)
        circle[mask] = 1
        return circle
    
    return circle_mask


def mrec_mask(image_shape: Tuple[int, int]) -> \
        Callable[[int, int, int, int], np.ndarray]:
    """
    Defines variables used inside rec_mask
    """
    x = np.arange(image_shape[0]).reshape(1,-1)
    y = np.arange(image_shape[1]).reshape(-1,1)
    img_shape = image_shape

    @lru_cache(maxsize=1024)
    def rec_mask(cx: int, cy: int, 
            w: int, h: int) -> np.ndarray:
        """
        Creates a circular mask at the given center
        coordinates
        """
        mask = ((x + w > cx) & (x - w < cx)) & \
            ((y + h > cy) & (y - h < cy))
        rec = np.zeros(img_shape, dtype=np.int8)
        rec[mask] = 1
        return rec

    return rec_mask


def overlay(_img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Overlays img2 into img1
    """
    img1 = _img1.copy() if _img1.shape == img2.shape \
        else _img1.reshape(img2.shape)

    mask = img2[:,:,-1] >= 200
    img1[mask] = img2[mask]
    return img1