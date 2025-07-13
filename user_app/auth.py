from .lib.db_helper import *  
from flask import Blueprint, redirect, render_template, request, session, url_for, current_app, jsonify, send_file


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
        return redirect(url_for("auth.login", username=username))
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
    return redirect(url_for("auth.closet", username=user[1], userid=user[0]))


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


@auth.route('/closet', methods=['GET',])
def closet():
    username = request.args.get('username')
    userid = request.args.get("userid")
    current_app.logger.info('Getting user') 
    image_s3_uris = get_user_apparels(userid)
    images = [uri[0] for uri in image_s3_uris]
    return render_template("closet.html", closet=images, userid=userid, username=username) 
