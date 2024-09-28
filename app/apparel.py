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
@apparel.route('/closet/<userid>', methods=('GET', 'POST'))
def closet(userid):
    if request.method == 'GET': 
        try:
            username = 'hanishadatwani'
            # susername = request.form['username']
        except KeyError:            
            return "422 USERNAME OR PASSWORD NOT SUBMITTED"        
        if username:
            userid = get_user(username)
            if get_apparel(userid):            
                return get_apparel(userid)
            else:            
                return "SOMETHING WENT WRONG"
    else: 
        try:
            username = request.form['username']
            password = request.form['password']
        except KeyError:
            return "422 USERNAME OR PASSWORD NOT SUBMITTED"
        if username and password:
            items = list()
            return items
    return render_template("index.html")