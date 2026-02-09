import mysql.connector, os, logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "closetx"),
    "password": os.getenv("DB_PASS", "hello"),
    "database": os.getenv("DB_NAME", "closetx"),
    "port": int(os.getenv("DB_PORT", 3306))
}

logger.info("db_config_loaded", extra={"config": {k: ("****" if k=="password" else v) for k,v in DB_CONFIG.items()}})

def get_conn():
    conn = mysql.connector.connect(**DB_CONFIG)
    logger.info("db_connection_opened")
    return conn