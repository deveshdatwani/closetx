import os
import sys
import torch
import logging  
from flask import Flask
from . import serve_model
sys.path.append("/home/deveshdatwani/closetx/models")
from models.apparel_encoder_models.model import EfficientNet

def create_app(config_file=None): 
    app = Flask(__name__)
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    app.logger = logging.getLogger("mlapp-logger")    
    app.logger.setLevel(logging.INFO) 
    if not config_file:
        app.logger.info("Spinning ml-app")
        app.logger.warning("No config file found")
    app.matcher = EfficientNet()
    app.register_blueprint(serve_model.serve_model)
    return app