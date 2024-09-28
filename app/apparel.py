import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app, send_file
from .lib.db_helper import * 


apparel = Blueprint("apparel", __name__)


# add or get apparel in closet
@apparel.route('/closet', methods=('GET', 'POST'))
def closet():
    uri = "./image-3.png"
    if request.method == 'POST': 
        try:
            username = request.form['username']
        except KeyError:            
            return "422 USERNAME NOT SUBMITTED"        
        userid = get_user(username)
        if post_apparel(userid[0][0], uri):            
            return "200"
        else:            
            return "SOMETHING WENT WRONG"
    else: 
        try:
            username = request.form['username']
        except KeyError:
            return "422 USERNAME OR PASSWORD NOT SUBMITTED"
        userid = get_user(username)
        apparel = get_apparel(userid[0][0])
        file_names = [i[1] for i in apparel] 
        images = list()
        for file_name in file_names:
            images.append(get_images(file_name))
        return images