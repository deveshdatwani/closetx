import requests
import functools
from lib.db_helper import * 
from lib.error_codes import ResponseString 
from flask import Blueprint, g, redirect, render_template, request, session, url_for, current_app, jsonify


auth = Blueprint("auth", __name__)
response_string = ResponseString()


@auth.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        data = "Forbidden"
        return serve_response(data, 403)
    else:
        return render_template('index.html')


@auth.route('/register', methods=['POST',])
def register():
    try:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
    except KeyError:
        current_app.logger.error("Missing request parameters")
        data = "Missing request parameters"           
        return serve_response(data, 422)        
    if username and password and email:
        current_app.logger.info("Registering user")
        if register_user(username, password, email):
            data = "User registered successfully" 
            return serve_response(data, 200)
        else:
            data = "Something went wrong"            
            return serve_response(data, 403)
    

@auth.route('/login', methods=['POST',])
def login():
    try:
        username = request.form['username']
        password = request.form['password']
    except KeyError:
        current_app.logger.error("Missing request parameters")
        data = "Missing request parameters"             
        return serve_response(data, 422)   
    user = login_user(username, password)
    if user:
        data = {"message":"Login success", "details": user[:3]}
        return jsonify(data)
    elif user == "Incorrect password": 
        data = user
        return serve_response(data, 201)
    else:
        data = "Something went wrong"
        return serve_response(data, status_code=504)       


@auth.route('/logout', methods=['DELETE',])
def logout():
    session.clear() 
    return redirect(url_for('index'))


@auth.route('/delete', methods=['DELETE',])
def delete():
    username = request.form['username']
    if delete_user(username): 
        data = "User deleted"           
        return serve_response(data, status_code=200)
    else:
        data = "Something went wrong"      
        return serve_response(data=data, status_code=500)


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