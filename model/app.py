import os
from flask import Flask
from . import ml
import logging  


def create_app(config_file=None): 
    app = Flask(__name__)
    app.logger = logging.Logger("INITIALIZING CLOSETX MODEL LOGGER")    
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("ML APPLICATION CONFIGURED SUCCESSFULLY")
        except Exception as e:
            app.logger.error(f"CORRUPT CONFIG FILE {e}")
    else:
        app.logger.warn("NO CONFIG FILE FOUND")
    app.register_blueprint(ml.ml_model)
    return app