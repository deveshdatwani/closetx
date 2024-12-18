import functools
import jwt
from .lib.db_helper import * 
from .lib.img_utils import *
from flask import Blueprint, g, request, session, current_app, send_file, jsonify, url_for, redirect


apparel = Blueprint("apparel", __name__)


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        token = request.headers["JWT-header"]
        try:
            decoded = jwt.decode(token, key="closetx_secret")
        except jwt.InvalidTokenError:
            current_app.logger.warning("Not authorized to the resource. Log in first")
            return redirect(url_for('auth.index'))
        except jwt.InvalidSignatureError:
            current_app.logger.warning("Inoccrect signature key")
            return redirect(url_for('auth.index'))
        else:
            return view()
    return wrapped_view


@apparel.route('/closet', methods=['POST',])
def add_apparel():
    try:
        userid = request.form['userid']
        image_file = request.files['image']
    except KeyError:
        current_app.logger.error("Missing request parameters")              
        return serve_response(data="Missing request parameters", status_code=403)
    if post_apparel(userid, image_file):           
        return serve_response(data="Image added", status_code=201)
    else:            
        return serve_response(data="Something went wrong", status_code=404)
    

@apparel.route('/closet/apparel', methods=['GET',])
def get_use_apparel():
    try:
        image_uri = request.form['uri']
    except KeyError:    
        current_app.logger.error("Missing request parameters")        
        return serve_response(data="Missing reques parameters", status_code=403)
    apparel_image = get_apparel(image_uri)
    if not apparel_image: return serve_response(data="No apparel found", status_code=404)
    else: return send_file(apparel_image, mimetype='image/png')
    

@apparel.route('/closet', methods=['GET',])
def get_user_closet():
    try:
        userid = request.form['userid']
    except KeyError:
        current_app.logger.error("Missing request parameters") 
        return serve_response(data="Missing reques parameters", status_code=403)
    apparel_ids = get_user_apparels(userid)
    print(apparel_ids)
    data = jsonify({"apparels" : apparel_ids})
    return data


@apparel.route('/closet', methods=['DELETE',])
def remove_apparel():
    try:
        userid = request.form['userid']
        uri = request.form['uri']
    except KeyError:
        current_app.logger.error("Missing request parameters")              
        return serve_response(data="Missing request parameters", status_code=403)
    response = delete_apparel(userid, uri)
    if response: data = "Apparel deleted successfully"
    else: data = "Something went wrong"
    return serve_response(data=data, status_code=203)
