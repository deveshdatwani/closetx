import os
import sys
import torch
import logging  
from flask import Flask
from pathlib import Path
from . import serve_model


def create_app(config_file=None): 
    app = Flask(__name__)
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    app.logger = logging.getLogger("mlapp-logger")    
    app.logger.setLevel(logging.INFO) 
    from models.apparel_encoder_models.model import EfficientNet
    app.register_blueprint(serve_model.serve_model)
    app.matcher = EfficientNet()
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("Application configured succesfully from config file")
        except Exception as e:
            app.logger.error(f"Corrupt config file")
    else:
        app.logger.warning("No config file found") 
        app.config["access_key"] = os.environ.get("AWS_ACCESS_KEY", default=None)
        app.config["secret_key"] = os.environ.get("AWS_SECRET_KEY", default=None)
    return app
