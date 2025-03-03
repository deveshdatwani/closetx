import functools
import jwt
import json
from .lib.db_helper import * 
from flask import Blueprint, g, request, session, current_app, send_file, jsonify, url_for, redirect


apparel = Blueprint("apparel", __name__, url_prefix="/closet_app")


@apparel.route('/closet/apparel', methods=['GET',])
def get_apparel_image():
    image_uri = request.form['uri']  
    current_app.logger.info("Getting apparel")
    apparel_image = get_apparel(image_uri)
    return send_file(apparel_image, mimetype='image/png')


@apparel.route('/closet', methods=['POST',])
def add_apparel():
    userid = request.form['userid']
    image_file = request.files['image']
    post_apparel(userid, image_file)           
    return serve_response(data="Apparel added", status_code=201)


@apparel.route('/add-apparel', methods=['POST'])
def add_session_apparel():
    apparel_img = request.files['image']
    apparel_id = request.form['apparel_id']
    colors = dict(rgb_from_img(apparel_img).json())
    img = Image.open(apparel_img)
    img.save(os.path.join(current_app.config['CACHE_DIR'], apparel_id + '.png'), 'png')
    colors["apparel_id"] = apparel_id
    current_app.config['global_dict'][apparel_id] = colors
    return jsonify(colors)


@apparel.route('/get-cached-apparel', methods=['POST', 'GET'])
def get_cached_apparel():
    img_id = request.form['apparel_id']
    img = cached_apparel(str(img_id))
    print(current_app.config['global_dict'])
    return send_file(img, mimetype='image/png')


@apparel.route('/match-with-all-cache', methods=['POST', 'GET'])
def match_with_all_cache():
    img = request.files['image']
    response = match_all(img, current_app.config['global_dict'])
    return jsonify(response)


#-----
    

@apparel.route('/closet', methods=['GET',])
def get_user_closet():
    userid = request.form['userid']
    current_app.logger.info("Getting closet apparels")
    apparel_ids = get_user_apparels(userid)
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
    else: return serve_response(data="Something went wrong", status_code=203)


@apparel.route('/delete/all', methods=['DELETE',])
def remove_closet():
    try:
        userid = request.form['userid']
    except KeyError:
        current_app.logger.error("Missing request parameters")              
        return serve_response(data="Missing request parameters", status_code=403)
    response = delete_closet(userid)
    if response: data = "Closet deleted successfully"
    else: return serve_response(data="Something went wrong", status_code=203)