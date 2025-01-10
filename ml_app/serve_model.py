import os
from PIL import Image
from io import BytesIO
from models.encoder.utils import *
from flask import Blueprint, request, current_app
from models.encoder.utils import get_median_pixel
from models.encoder.color_encoder import palette_rbg_list as palette

serve_model = Blueprint("serve_model", __name__, url_prefix="/model")


@serve_model.route("/")
def index():
    return "Closetx says, Hi"


@serve_model.route("/match", methods=['POST',])
def match():
    top_image = request.files["top"]
    bottom_image = request.files["bottom"]
    top_color, bottom_color = get_outfit_colors(top_image, bottom_image)
    if bottom_color in palette[top_color]: 
        return "Match 100%"
    else:
        return "Match 0%"