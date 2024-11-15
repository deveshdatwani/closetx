import os
from flask import Flask
import auth, apparel
import logging  
from lib.error_codes import ResponseString 


def create_app(config_file=None): 
    app = Flask(__name__)
    app.loggerlogger = logging.getLogger('my_logger')
    app.logger.setLevel(logging.INFO)
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("APPLICATION CONFIGURED SUCCESSFULLY")
        except Exception as e:
            app.logger.error(f"CORRUPT CONFIG FILE {e}")
    else:
        app.logger.warning("NO CONFIG FILE FOUND") 
        app.apparel_folder = "./apparel"
    app.register_blueprint(auth.auth)
    app.register_blueprint(apparel.apparel)
    app.access_key = os.environ.get("ACCESS_KEY", default=None)
    app.secret_key = os.environ.get("SECRET_KEY", default=None)
    app.error_codes = ResponseString()
    return app