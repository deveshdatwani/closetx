import mysql.connector, os

DB = {
    "host": os.getenv("DB_HOST","localhost"),
    "user": os.getenv("DB_USER","closetx"),
    "password": os.getenv("DB_PASS","hello"),
    "database": os.getenv("DB_NAME","closetx")
}

def get_conn():
    return mysql.connector.connect(**DB)

