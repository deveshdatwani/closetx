import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app, send_file
from lib.db_helper import * 
from lib.img_utils import *


apparel = Blueprint("apparel", __name__)


@apparel.route('/closet/user/add_apparel', methods=['POST',])
def add_apparel():
    try:
        userid = request.form['userid']
        image_file = request.files['image']
    except KeyError:            
        return current_app.error_codes.no_username_or_password 
    image = watershed_segmentation(image_file)
    if post_apparel(userid, image):           
        return serve_response(data="IMAGE ADDED", status_code=201)
    else:            
        return current_app.error_codes.something_went_wrong
    

@apparel.route('/closet/user/get_apparel', methods=['POST',])
def get_apparels():
    try:
        image_uri = request.form['uri']
    except KeyError:    
        current_app.logger.error("MISSING REQUEST PARAMETERS")        
        return current_app.error_codes.no_username_or_password 
    apparel_image = get_apparel(image_uri)
    if not apparel_image: return serve_response(data="NO SUCH APPAREL", status_code=404)
    else: return send_file(apparel_image, mimetype='image/png')
    

@apparel.route('/closet/user/all_apparel', methods=['POST',])
def user_closet():
    userid = request.form['userid']
    apparel_ids = get_user_apparels(userid)
    print(apparel_ids)
    return apparel_ids