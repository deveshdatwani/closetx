import os
import io
from PIL import Image
from io import BytesIO
from ml_app.lib.utils import *
from flask import Blueprint, request, current_app, send_file
from .models.encoder.color_encoder import get_palette_color
from .models.encoder.color_encoder import palette_rbg_list as palette


serve_model = Blueprint("serve_model", __name__, url_prefix="/model")


@serve_model.route("/")
def index():
    return "Closetx model says, Hi"


@serve_model.route("/segment", methods=['POST'])
def segment():
    image = request.files['image']
    segmented_image = seg_apparel(image, current_app.segmentation_model)
    img_io = return_segmented_image(segmented_image)
    return send_file(img_io, mimetype="image/png")


@serve_model.route("/color", methods=['POST',])
def get_color():
    r, g, b = int(request.form['r']), int(request.form['g']), int(request.form['b'])
    palette_color = get_palette_color((r,g,b))
    return palette_color


# -------------- #


@serve_model.route("/match", methods=['POST',])
def match_apparels():
    top_image = request.files["top"]
    bottom_image = request.files["bottom"]
    top_color, bottom_color = get_outfit_colors(top_image, bottom_image, current_app.segmentation_model)
    match = get_palette_color(top_color, bottom_color)
    return str(match)


@serve_model.route("/match/raw", methods=['POST',])
def match_raw():
    apparel = request.files["apparel"]
    apparel = Image.open(BytesIO(apparel.read())).convert("RGB")
    seg_apparel_img = seg_apparel(apparel, current_app.segmentation_model)
    raw_match(seg_apparel_img)
    return "Done"