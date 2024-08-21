from flask import g, current_app
import click
import sqlite3


@click.command('init-db')
def init_app_db():
    db = sqlite3.connect(current_app.config['DATABASE'], 
                         detect_types=sqlite3.PARSE_DECLTYPES)
    with current_app.open_resource('/home/deveshdatwani/closetx/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    db.close()
    
    return 1