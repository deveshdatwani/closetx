import os
from flask import Flask
import logging  
from . import serve_model


def create_app(config_file=None): 
    app = Flask(__name__)
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    app.logger = logging.getLogger("mlapp-logger")    
    app.logger.setLevel(logging.INFO) 
    app.register_blueprint(serve_model.serve_model)
    return app