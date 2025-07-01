from .lib.db_helper import *  
from user_app.celery_app import add
from flask import Blueprint, redirect, render_template, request, session, url_for, current_app, jsonify


auth = Blueprint("auth", __name__, url_prefix="/auth")


@auth.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@auth.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        current_app.logger.info("Registering user")
        register_user(username, password)
        return redirect(url_for("auth.user", username=username))
    else:
        return render_template("register.html")
    

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET': return render_template("login.html")
    username = request.form['username']
    password = request.form['password']
    current_app.logger.info("Logging in user")
    user = login_user(username, password)
    if not user: return redirect(url_for("auth.login"))
    return jsonify(user)


@auth.route('/logout', methods=['DELETE',])
def logout():
    userid = request.form['userid']
    session.clear() 
    return redirect(url_for('index'))


@auth.route('/delete', methods=['DELETE',])
def delete():
    username = request.form['username']
    current_app.logger.info('Deleting user')    
    delete_user(username) 
    return serve_response(data="User deleted", status_code=200)


@auth.route('/user', methods=['GET',])
def user():
    username = request.args.get('username')
    user = get_user(username)
    current_app.logger.info('Getting user')
    return jsonify(user)


@auth.route("/task/<int:number1>/<int:number2>", methods=["GET", "POST"])
def add_task(number1, number2):
    return str(add.delay(number1, number2))