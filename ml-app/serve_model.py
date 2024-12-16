import io
import cv2
import torch
import functools
import numpy as np
from .lib.db_helper import *
from matplotlib import pyplot as plt  
from flask import Blueprint, g, request, redirect, current_app


serve_model = Blueprint("serve_model", __name__)


@serve_model.route('/', methods=['GET',])
def index():
    return "Invalid URL"


@serve_model.route('/apparel/match', methods=['GET', 'POST'])
def match_apparel():
    top = request.files['top']
    top = Image.open(top)
    top = np.array(top)
    top = cv2.resize(top,(576, 576))
    top = cv2.cvtColor(np.array(top), cv2.COLOR_BGR2RGB)
    bottom = request.files['bottom']
    bottom = Image.open(bottom)
    bottom = np.array(bottom)
    bottom = cv2.resize(bottom,(576, 576))
    bottom = cv2.cvtColor(np.array(bottom), cv2.COLOR_BGR2RGB)
    output = inference(current_app.matcher, top, bottom)
    return f"score {output.item()}"
    return redirect("http://localhost:5500/alpha.html")