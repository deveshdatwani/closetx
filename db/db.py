import click
import sqlite3
from logging import Logger
from flask import current_app, g
from sqlalchemy import create_engine
log = Logger('db logger')


@click.command('init-db')
def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    
    return "SUCCESS " 


def get_db_connector():
    if 'db' not in g:
        db_config = current_app.config['DATABASE'] 
        print(db_config)
        
        return None


def get_db():
    if 'db' not in g:    
        if current_app.config['DATABASE']:
            g.db = sqlite3.connect(
                current_app.config['DATABASE'],
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
        else:
            log.warn("NO DATABASE CONFIG FOUND")

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()