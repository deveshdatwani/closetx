import os
from flask import Flask
from . import auth, apparel
import logging  


def create_app(config_file=None): 
    app = Flask(__name__)
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    app.logger = logging.getLogger("closetx-logger")    
    app.logger.setLevel(logging.INFO)
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("APPLICATION CONFIGURED SUCCESSFULLY")
        except Exception as e:
            app.logger.error(f"CORRUPT CONFIG FILE {e}")
    else:
        app.logger.warn("NO CONFIG FILE FOUND") 
        app.apparel_folder = "./apparel"
    app.register_blueprint(auth.auth)
    app.register_blueprint(apparel.apparel)
    return app