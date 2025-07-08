from .lib.db_helper import * 
from flask import Blueprint, request, current_app, send_file, jsonify, redirect, url_for
from . import auth

apparel = Blueprint("apparel", __name__, url_prefix="/closet")


@apparel.route('/post/apparel', methods=['POST',])
def add_apparel():
    userid = request.form['userid']
    image_file = request.files['image']
    image_file = correct_image_orientation(image_file)
    post_apparel(userid, image_file)           
    return redirect(url_for("auth.closet", userid=userid))


@apparel.route('/apparel', methods=['DELETE',])
def remove_apparel():
    uri = request.form['uri']
    delete_apparel(uri)
    current_app.logger.info(f"Delete apparel {uri}")
    return 203


@apparel.route('/image_proxy', methods=['GET', 'POST'])
def image_proxy():
    uri = request.args.get('uri')
    img_file = fetch_image_base64(uri)
    img_file.seek(0)
    return send_file(img_file, mimetype='image/png')