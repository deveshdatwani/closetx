import os
from flask import Flask
from . import auth
from .init_db import init_app_db


def create_app(config_file=None, db=None):
    app = Flask(__name__)

    if config_file:
        app.config.from_file(config_file)
    
    else:
        app.config['DATABASE'] = os.path.join(app.instance_path, 'closetx.sqlite')

    if db:
        db.init()

    app.register_blueprint(auth.auth)
    app.cli.add_command(init_app_db)
    
    return app