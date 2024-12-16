import os
import logging  
import auth, apparel
from flask import Flask
from lib.error_codes import ResponseString 
from prometheus_flask_exporter import PrometheusMetrics


def create_app(config_file=None): 
    app = Flask(__name__)
    app.loggerlogger = logging.getLogger('my_logger')
    app.logger.setLevel(logging.INFO)
    app.config["SECRET"] = "closetx_secret"
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
    app.register_blueprint(auth.auth)
    app.register_blueprint(apparel.apparel)
    return app
