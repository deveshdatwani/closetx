import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app, send_file


ml_model = Blueprint("inference", __name__)


# add or get apparel in closet
@ml_model.route('/forward', methods=('GET', 'POST'))
def closet():
    if request.method == 'POST':             
        return "SOMETHING WENT WRONG"
    elif request.method == "GET": 
        return "NOTHING"