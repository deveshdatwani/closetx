import requests
import functools
from lib.db_helper import * 
from lib.error_codes import ResponseString 
from flask import Blueprint, g, redirect, render_template, request, session, url_for, current_app, jsonify


auth = Blueprint("auth", __name__)
response_string = ResponseString()


@auth.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@auth.route('/register', methods=['POST',])
def register():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']     
    if username and password and email:
        current_app.logger.info("Registering user to app")
        if register_user(username, password, email):
            data = "User registered successfully" 
            return serve_response(data, 200)
        else:
            data = "Something went wrong"            
            return serve_response(data, 403)
    else:
        return serve_response(data="Something went wrong", status_code=400)
    

@auth.route('/login', methods=['POST',])
def login():
    username = request.form['username']
    password = request.form['password'] 
    user = login_user(username, password)
    if user:
        data = {"message":"Login success", "details": user[:3]}
        return jsonify(data)
    elif user == "Incorrect password": 
        data = user
        return serve_response(data, 200)       


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