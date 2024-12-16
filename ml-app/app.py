import os
import sys
import torch
import logging  
from flask import Flask
from . import serve_model


def create_app(config_file=None): 
    app = Flask(__name__)
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    app.logger = logging.getLogger("mlapp-logger")    
    app.logger.setLevel(logging.INFO) 
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("Application configured succesfully from config file")
        except Exception as e:
            app.logger.error(f"Corrupt config file")
    else:
        app.logger.warning("No config file found") 
        app.access_key = os.environ.get("ACCESS_KEY", default=None)
        app.secret_key = os.environ.get("SECRET_KEY", default=None)
    sys.path.append("/home/deveshdatwani/closetx/models")
    from models.apparel_encoder_models.model import EfficientNet
    app.matcher = EfficientNet()
    app.register_blueprint(serve_model.serve_model)
    
    @app.route('/', methods=['GET',])
    def index():
        return "Invalid URL"
    
    return app