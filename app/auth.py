import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from .lib.db_helper import * 


auth = Blueprint("auth", __name__)


@auth.route('/')
def index():
    if request.method == "POST":
        
        return "405"
    else:
        
        return render_template("index.html")


# Binds a URL to the app for requesting the register template and also signing up a user
@auth.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST': 
        try:
            username = request.form['username']
            password = request.form['password']
        except KeyError:
            
            return "422 USERNAME OR PASSWORD NOT SUBMITTED"
        
        if username and password:
            if register_user(username, password):
            
                return "200 SUCCESSFULLY REGISTERED USER"
            else:
            
                return "SOMETHING WENT WRONG"

    return render_template("index.html")
    

# Registers a login page
@auth.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username and password:
            if login_user(username, password) == "CORRECT PASSWORD":
            
                return "LOGIN SUCCESS"
            else: 
            
                return "INCORRECT PASSWORD"      
        else: 
            
            return "USERNAME AND PASSWORD NOT SUBMITTED"

    return "GET WELCOME TO CLOSETX"


# Logout and clear session 
@auth.route('/logout')
def logout():
    session.clear()
    
    return redirect(url_for('index'))


# Delete user account
@auth.route('/delete', methods=('POST', 'GET'))
def delete():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if delete_user(username,):
            
            return "USER DELETED SUCCESSFULLY"
        else:
            
            return "SOMETHING WENT WRONG"


# Check whether the user has an active session
@auth.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db_x().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


# Authenticate user is logged in 
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view