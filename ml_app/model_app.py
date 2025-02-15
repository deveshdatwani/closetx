import os
import io
from PIL import Image
from io import BytesIO
from ml_app.lib.utils import *
from flask import Blueprint, request, current_app, send_file
from .models.encoder.color_encoder import get_palette_color
from .models.encoder.color_encoder import palette_rbg_list as palette, match_apparel_color


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


@serve_model.route("/match-color")
def match_color():
    r1, g1, b1 = int(request.form['r1']), int(request.form['g1']), int(request.form['b1'])
    r2, g2, b2 = int(request.form['r2']), int(request.form['g2']), int(request.form['b2'])
    match_result = match_apparel_color(r1,g1,b1,r2,g2,b2)
    return match_result
# -------------- #

@serve_model.route("/match", methods=['POST',])
def match_apparels():
    top_image = request.files["top"]
    bottom_image = request.files["bottom"]
    top_color, bottom_color = get_outfit_colors(top_image, bottom_image, current_app.segmentation_model)
    match = get_palette_color(top_color, bottom_color)
    return str(match)