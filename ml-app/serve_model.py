import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from .lib.db_helper import * 
from .lib.error_codes import ResponseString 


serve_model = Blueprint("serve_model", __name__)
response_string = ResponseString()


# register user
@serve_model.route('/')
def index():
    return "HELLO"