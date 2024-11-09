import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from .lib.db_helper import * 
from .lib.error_codes import ResponseString 
import requests

auth = Blueprint("auth", __name__)
response_string = ResponseString()


# index
@auth.route('/')
def index():
    if request.method == "POST":
        return "405"
    else:
        return current_app.secret_key
        return render_template("index.html")


# register user
@auth.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        print("POST REQUEST") 
        try:
            username = request.form['username']
            password = request.form['password']
        except KeyError:            
            return "422 USERNAME OR PASSWORD NOT SUBMITTED"        
        if username and password:
            print("TRYING TO REGISTER USER")
            if register_user(username, password):            
                return "200 SUCCESSFULLY REGISTERED USER"
            else:            
                return "SOMETHING WENT WRONG"
    return render_template("index.html")
    

# login user
@auth.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        current_app.logger.info("username and password found")
        if username and password:
            if login_user(username, password) == "CORRECT PASSWORD":
                return "LOGIN SUCCESS"
            else: 
                return "INCORRECT PASSWORD"      
        else: 
            return "USERNAME AND PASSWORD NOT SUBMITTED"
    return "GET WELCOME TO CLOSETX"


# logout user
@auth.route('/logout')
def logout():
    session.clear() 
    return redirect(url_for('index'))


# delete user
@auth.route('/delete', methods=('POST', 'GET'))
def delete():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if delete_user(username,):            
            return "USER DELETED SUCCESSFULLY"
        else:      
            return "SOMETHING WENT WRONG"


# validate user session
@auth.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = get_db_x().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


# validate user login 
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:            
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view