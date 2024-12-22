from .lib.db_helper import *  
from flask import Blueprint, g, request, redirect, current_app


serve_model = Blueprint("serve_model", __name__)


@serve_model.route('/model', methods=['GET',])
def index():
    return "Welcome to closetx engine"


@serve_model.route('/model/apparel/match', methods=['GET', 'POST'])
def match_apparel():
    top = request.files['top']
    bottom = request.files['bottom']
    output = inference(current_app.match_engine, top, bottom)
    return f"score {output.item()}"
    return redirect("http://localhost:5500/alpha.html")