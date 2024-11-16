import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from lib.db_helper import * 
from lib.error_codes import ResponseString 
import requests

auth = Blueprint("auth", __name__)
response_string = ResponseString()


@auth.route('/', methods=('GET', 'POST'))
def index():
    if request.method == "POST":
        data = current_app.error_codes.forbidden
        return serve_response(data, 403)
    else:
        return render_template('index.html')


@auth.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
        except KeyError:
            data = current_app.error_codes.no_username_or_password            
            return serve_response(data, 422)        
        if username and password:
            current_app.logger.info("AUTHENTICATING USER")
            if register_user(username, password):
                data = current_app.error_codes.registered_user_successfully            
                return serve_response(data, 200)
            else:            
                return serve_response(current_app.error_codes.something_went_wrong, 403)
    return render_template("index.html")
    

@auth.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
        except KeyError:
            data = current_app.error_codes.no_username_or_password            
            return serve_response(data, 422)   
        if login_user(username, password):
            data = current_app.error_codes.login_success
            return serve_response(data, 200)
        else: 
            data = current_app.error_codes.incorrect_password
            return serve_response(data, 200)       
    return redirect('/')


@auth.route('/logout')
def logout():
    session.clear() 
    return redirect(url_for('index'))


@auth.route('/delete', methods=('POST', 'GET'))
def delete():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if delete_user(username,):            
            return "USER DELETED SUCCESSFULLY"
        else:      
            return "SOMETHING WENT WRONG"


@auth.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = get_db_x().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:            
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view