import mysql.connector, os

DB_CONFIG = {"host": os.environ.get("DB_HOST","localhost"), "user": os.environ.get("DB_USER","user"), "password": os.environ.get("DB_PASS","pass"), "database": os.environ.get("DB_NAME","db")}

print(DB_CONFIG)

def get_conn():
    return mysql.connector.connect(**DB_CONFIG)
