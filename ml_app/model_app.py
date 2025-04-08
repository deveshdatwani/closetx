from .lib.utils import *
from flask import Blueprint, request, current_app, send_file, jsonify
from .models.encoder.color_encoder import *
# from .models.encoder.color_encoder import 


serve_model = Blueprint("model_app", __name__, url_prefix="/model")


@serve_model.route("/")
def index():
    return "Closetx model says, Hi"


@serve_model.route("/segment", methods=['POST'])
def segment():
    image = request.files['image']
    segmented_image = seg_apparel(image, current_app.segmentation_model)
    img_io = return_segmented_image(segmented_image)
    return send_file(img_io, mimetype="image/png")


@serve_model.route("/get-color", methods=['POST',])
def get_color():
    r, g, b = int(request.form['r']), int(request.form['g']), int(request.form['b'])
    palette_color = match_colors(match_color=(r,g,b))
    return palette_names[palette_color]


@serve_model.route("/match", methods=['POST',])
def match_color():
    r1, g1, b1 = int(request.form['r1']), int(request.form['g1']), int(request.form['b1'])
    r2, g2, b2 = int(request.form['r2']), int(request.form['g2']), int(request.form['b2'])
    match_result = match_apparel_color(r1,g1,b1,r2,g2,b2)
    return str(match_result)


@serve_model.route("/get-color-from-apparel", methods=['POST',])
def get_color_from_apparel():
    image = request.files['image']
    segmented_image = seg_apparel(image, current_app.segmentation_model)
    plt.imshow(segmented_image)
    plt.show()
    median_r, median_g, median_b = get_median_pixel(segmented_image)
    color = match_colors(match_color=(median_r, median_g, median_b))
    return jsonify({"color":color, "r":median_r, "g":median_g, "b":median_b})


@serve_model.route("/classify", methods=['POST',])
def classify_apparel():
    image = request.files['image']
    label = classify_from_image(image, current_app.segmentation_model)
    return label