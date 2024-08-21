import os
from flask import Flask
from . import auth
from .init_db import init_app_db
import logging  


def create_app(config_file=None): 
    app = Flask(__name__)
    app.logger = logging.Logger("closetx base logger")    

    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("APPLICATION CONFIGURED SUCCESSFULLY")
        except Exception as e:
            app.logger.error(f"CORRUPT CONFIG FILE {e}")
    
    else:
        app.logger.warn("NO CONFIG FILE FOUND") 

    app.register_blueprint(auth.auth)
    
    return app