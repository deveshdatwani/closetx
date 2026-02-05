from flask import Flask
import torch
import os

def create_app(config_name=None):
    app = Flask(__name__)
    if config_name:
        app.config.from_object(config_name)
    from .routes.endpoint1 import bp as ep1_bp
    app.register_blueprint(ep1_bp, url_prefix="/ep1")
    from .routes.endpoint2 import bp as ep2_bp
    app.register_blueprint(ep2_bp, url_prefix="/ep2")
    return app

