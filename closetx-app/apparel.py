import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app, send_file
from lib.db_helper import * 
from lib.img_utils import *


apparel = Blueprint("apparel", __name__)


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
    return apparel_image
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
    data = jsonify({"apparels" : apparel_ids})
    return data