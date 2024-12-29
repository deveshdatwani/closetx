import jwt
from .lib.db_helper import *  
from flask import Blueprint, g, redirect, render_template, request, session, url_for, current_app, jsonify


auth = Blueprint("auth", __name__, url_prefix="/app")


@auth.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@auth.route('/register', methods=['POST',])
def register():
    try:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
    except KeyError:
        current_app.logger.error("Missing request parameters")
        return serve_response(data="Missing request parameters" , status_code=422)        
    current_app.logger.info("Registering user")
    if register_user(username, password, email):
        return serve_response(data="User registered successfully" , status_code=200)
    else:            
        return serve_response(data="Something went wrong", status_code=403)
    

@auth.route('/login', methods=['POST',])
def login():
    try:
        username = request.form['username']
        password = request.form['password']
    except KeyError:
        current_app.logger.error("Missing request parameters")             
        return serve_response(data="Missing request parameters" , status_code=422)   
    user = login_user(username, password)
    if user:
        data = jsonify({"message":"Login success", "user_details": user[:3]})
        data.headers["JWT-header"] = jwt.encode(payload={"user":user[:3]}, key="closetx_secret", algorithm='HS256')
        return data
    elif user == "Incorrect password": 
        return serve_response(data=user, status_code=201)
    elif not user:
        return serve_response(data="Could not find user with given username", status_code=503)       


@auth.route('/logout', methods=['DELETE',])
def logout():
    session.clear() 
    return redirect(url_for('index'))


@auth.route('/delete', methods=['DELETE',])
def delete():
    username = request.form['username']
    if delete_user(username):     
        return serve_response(data="User deleted", status_code=200)
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
