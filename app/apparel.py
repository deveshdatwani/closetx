import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from .lib.db_helper import * 


apparel = Blueprint("apparel", __name__)


@apparel.route('/')
def index():
    if request.method == "POST":
        
        return "405"
    else:        
        return render_template("index.html")


# Binds a URL to the app for requesting the register template and also signing up a user
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
        userid = get_user(username)[0][0]
        apparel = get_apparel(userid)
    return render_template("index.html")