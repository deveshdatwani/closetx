import os
from PIL import Image
from io import BytesIO
from ml_app.lib.utils import *
from flask import Blueprint, request, current_app
from ml_app.lib.utils import get_median_pixel
from .models.encoder.color_encoder import palette_rbg_list as palette


serve_model = Blueprint("serve_model", __name__, url_prefix="/ml_app")


@serve_model.route("/")
def index():
    return "Closetx model says Hi"


@serve_model.route("/segment")
def segment_apparel():
    image = request.form['image']


@serve_model.route("/match", methods=['POST',])
def match():
    top_image = request.files["top"]
    bottom_image = request.files["bottom"]
    top_color, bottom_color = get_outfit_colors(top_image, bottom_image, current_app.segmentation_model)
    match = get_match(top_color, bottom_color)
    return str(match)


@serve_model.route("/match/raw", methods=['POST',])
def match_raw():
    apparel = request.files["apparel"]
    apparel = Image.open(BytesIO(apparel.read())).convert("RGB")
    seg_apparel_img = seg_apparel(apparel, current_app.segmentation_model)
    raw_match(seg_apparel_img)
    return "Done"