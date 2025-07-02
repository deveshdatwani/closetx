from .lib.db_helper import * 
from flask import Blueprint, request, current_app, send_file, jsonify, redirect, url_for
from . import auth

apparel = Blueprint("apparel", __name__, url_prefix="/closet")


@apparel.route('/get/apparel', methods=['GET', "POST"])
def get_apparel_image():
    image_uri = request.form['uri']  
    current_app.logger.info("Getting apparel")
    apparel_image = get_apparel(image_uri)
    return send_file(apparel_image, mimetype='image/png')


@apparel.route('/post/apparel', methods=['POST',])
def add_apparel():
    userid = request.form['userid']
    image_file = request.files['image']
    post_apparel(userid, image_file)           
    return redirect(url_for("auth.closet", userid=userid))
 

@apparel.route('/closet/<string:uri>', methods=['GET',])
def get_user_closet(uri):
    data = get_apparel(uri)
    return send_file(data, mimetype='image/png')


@apparel.route('/apparel', methods=['DELETE',])
def remove_apparel():
    userid = request.form['userid']
    uri = request.form['uri']
    response = delete_apparel(userid, uri)
    return serve_response(data="Deleted apparel", status_code=203)


@apparel.route('/closet', methods=['DELETE',])
def remove_closet():
    userid = request.form['userid']
    response = delete_closet(userid)
    return serve_response(data="Closet deleted", status_code=203)