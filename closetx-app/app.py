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
    if config_file:
        try:
            app.config.from_file(config_file)
            app.logger.info("Application configured succesfully")
        except Exception as e:
            app.logger.error(f"Corrupt config file {e}")
    else:
        app.logger.warning("NO config file found") 
        app.apparel_folder = "./apparel"
        app.salt = "salt"
        app.pepper = "pepper"
    app.register_blueprint(auth.auth)
    app.register_blueprint(apparel.apparel)
    app.access_key = os.environ.get("ACCESS_KEY", default=None)
    app.secret_key = os.environ.get("SECRET_KEY", default=None)
    app.error_codes = ResponseString()
    metrics = PrometheusMetrics(app)
    return app
