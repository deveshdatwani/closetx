import jwt
from flask import redirect, url_for, current_app, request


def session_check(func):
    def wrapper():
        try:
            user = jwt.decode(request.form['token'], algorithms="HS256", key=current_app.config['secret'])
            return func()
        except jwt.exceptions.InvalidSignatureError:
            return redirect(url_for("auth.index"))
        except jwt.exceptions.DecodeError: 
            return redirect(url_for("auth.index"))
    return wrapper