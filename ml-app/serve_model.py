import io
import functools
from PIL import Image
from .lib.db_helper import *
from matplotlib import pyplot as plt 
from .lib.error_codes import ResponseString 
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app


serve_model = Blueprint("serve_model", __name__)
response_string = ResponseString()


# register user
@serve_model.route('/')
def index():
    return "HELLO"


@serve_model.route('/apparel/match', methods=['POST'])
def match_apparel():
    top = request.files['top']
    bottom = request.files['bottom']
    top = Image.open(io.BytesIO(top.read()))
    bottom = Image.open(io.BytesIO(bottom.read()))
    return "received images"