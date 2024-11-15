import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app, send_file
from lib.db_helper import * 


apparel = Blueprint("apparel", __name__)


# add or get apparel in closet
@apparel.route('/closet', methods=('GET', 'POST'))
def closet():
    if request.method == 'POST': 
        try:
            userid = request.form['userid']
            image_file = request.files['image']
        except KeyError:            
            return "USERID OR IMAGE NOT SUBMITTED"        
        if post_apparel(userid, image_file):            
            return "200 IMAGE ADDED"
        else:            
            return "NO IMAHE ADDED SOMETHING WENT WRONG"
    elif request.method == "GET": 
        try:
            userid = request.form['userid']
        except KeyError:
            return "422 USERID NOT SUBMITTED"
        apparel = get_apparel(userid)
        file_names = [i[1] for i in apparel] 
        images = list()
        for file_name in file_names:
            images.append(get_images(file_name))
        return images