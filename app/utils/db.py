import mysql.connector, os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

DB_CONFIG = {"host": os.environ.get("DB_HOST", "localhost"), "user": os.environ.get("DB_USER", "root"), "password": os.environ.get("DB_PASS", "hello"), "database": os.environ.get("DB_NAME", "closetx")}
logger.info(DB_CONFIG)

def get_conn():
    return mysql.connector.connect(**DB_CONFIG)
