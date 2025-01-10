import numpy as np
from PIL import Image
from io import BytesIO
from models.encoder.color_encoder import match_color


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