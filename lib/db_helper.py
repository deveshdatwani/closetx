from flask import session
from db.db import get_db
from werkzeug.security import check_password_hash, generate_password_hash


def register_user(username, password):
    db = get_db()
    db.execute("INSERT INTO user (username, password) VALUES (?, ?)",
                (username, generate_password_hash(password)))
    db.commit()
    db.close()


def login_user(username, password):
    db = get_db()
    user = db.execute("SELECT * FROM user WHERE username = ? ")
    
    if not user: return False
    elif check_password_hash(user["password"], password):
        session.clear()
        session['user_id'] = user['id']
        
        return True, user
        