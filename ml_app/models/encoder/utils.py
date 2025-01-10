import cv2
import numpy as np
from PIL import Image


def get_median_pixel(img: np.array) -> tuple:
    img_array = np.array(img)
    img_r = img_array[:, :, 0]
    median_r = img_r[img_r > 0]
    img_g = img_array[:, :, 1]
    median_g = img_g[img_g > 1]
    img_b = img_array[:, :, 2]
    median_b = img_b[img_b > 2]
    return median_r, median_g, median_b