from flask import Flask
from . import auth


def create_app(config_file=None, db=None):
    app = Flask(__name__)

    if config_file:
        app.config.from_file(config_file)

    if db:
        db.init()

    app.register_blueprint(auth.auth)
    
    return app