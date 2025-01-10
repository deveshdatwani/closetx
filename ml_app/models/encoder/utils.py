import logging
import numpy as np
from PIL import Image
from io import BytesIO
from models.encoder.color_encoder import palette_rgb as palette
from models.encoder.color_encoder import match_color
from models.encoder.color_encoder import palette_rbg_list as p_list


logger = logging.getLogger(__name__)

def get_median_pixel(img: np.array) -> tuple:
    img_array = np.array(img)
    img_r = img_array[:, :, 0]
    median_r = np.median(img_r[img_r > 0])
    img_g = img_array[:, :, 1]
    median_g = np.median(img_g[img_g > 1])
    img_b = img_array[:, :, 2]
    median_b = np.median(img_b[img_b > 2])
    return median_r, median_g, median_b


def get_outfit_colors(top_image, bottom_image):
    top_image = Image.open(BytesIO(top_image.read()))
    bottom_image = Image.open(BytesIO(bottom_image.read()))
    top_rgb = get_median_pixel(top_image)
    bottom_rgb = get_median_pixel(bottom_image)
    top_color = match_color(top_rgb)
    bottom_color = match_color(bottom_rgb)
    return top_color, bottom_color

def get_match(top_color, bottom_color):
    logger.info(f'{top_color} {bottom_color}')
    bottom = p_list[bottom_color]
    top = p_list[top_color]
    if bottom in palette[top]: return 100
    else: return 0