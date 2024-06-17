import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash
from flaskr.db import get_db
from lib.db_helper import *


bp = Blueprint("auth", __name__, url_prefix="/auth")


# Binds a URL to the app for requesting the register template and also signing up a user
@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username and password:
            register_user(username, password)
        else:
            redirect(url_for("auth.register"))

    return render_template('auth/register.html')
    

# Registers a login page
@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username and password:
            if login_user(username, password):
                return redirect(url_for("index"))
            else: redirect(url_for("auth.login"))
        else: redirect(url_for("auth.login"))

    return render_template('auth/login.html')


# Check whether the user has an active session
@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


# Logout and clear session 
@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# Authenticate user is logged in 
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view