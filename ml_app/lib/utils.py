import io
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from ..models.encoder.color_encoder import palette_rgb as palette
from ..models.encoder.color_encoder import get_palette_color as match_color
from ..models.encoder.color_encoder import palette_rbg_list as p_list
from ..models.huggingface_cloth_segmentation.process import *


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


def get_outfit_colors(top_image, bottom_image, model):
    top_image = Image.open(BytesIO(top_image.read())).convert('RGB')
    bottom_image = Image.open(BytesIO(bottom_image.read())).convert('RGB')
    top_image = seg_apparel(top_image, model, apparel_type=0)
    bottom_image = seg_apparel(bottom_image, model, apparel_type=1)
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


def seg_apparel(img, model, device='cpu', apparel_type=1):
    img = Image.open(img)
    palette = get_palette(4)
    masks, cloth_seg = generate_mask(img, net=model, palette=palette, device=device)
    apparel = cv2.bitwise_and(np.array(img), np.array(img), mask=np.array(masks[0], np.uint8))
    return apparel


def raw_match(img, closetx=None):
    closetx = [(243,198,189), (174,182,189), (200,223,236), (255,255,255), (26,72,113)]
    color = get_median_pixel(img)
    color = match_color(color)
    for closet in closetx:
        closet = match_color(closet)
        print(get_match(closet, color))    


def return_segmented_image(segmented_image):
    pil_image = Image.fromarray(segmented_image.astype(np.uint8))
    img_io  = io.BytesIO()
    pil_image.save(img_io, format="PNG")
    img_io.seek(0)
    return img_io


def match_apparel_color(r1,g1,b1,r2,g2,b2):
    match_result = "False"
    return match_result 