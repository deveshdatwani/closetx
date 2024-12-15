import io
import torch
import functools
import numpy as np
from .lib.db_helper import *
from matplotlib import pyplot as plt  
from flask import Blueprint, g, request, session, current_app


serve_model = Blueprint("serve_model", __name__)


@serve_model.route('/')
def index():
    return "Invalid URL"


@serve_model.route('/apparel/match', methods=['GET'])
def match_apparel():
    top = torch.rand(1, 3, 576, 576, dtype=torch.float32)
    bottom = torch.rand(1, 3, 576, 576, dtype=torch.float32)
    output = inference(current_app.matcher, top, bottom)
    return f"received images. output size {output}"