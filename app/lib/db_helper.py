from flask import session, g
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import check_password_hash, generate_password_hash
import logging


def get_db_x():
    try:
        cnx = mysql.connector.connect(
            user='closetx',
            password='password',
            host='127.0.0.1',
            database='closetx'
        )
        g.db = cnx
        print("MYSQL connector successfully connected to db")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Failed to authenicate MYSQL Connector -- something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        
        return None
    
    return cnx
    


def register_user(username, password):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            dbx.cursor().execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, password))
            dbx.commit()
            dbx.close()
        except mysql.connector.errors.IntegrityError:
            print("Duplicate entries")
            return False
        
        return True
    else:
        print("Could not connect") 
        return False


def login_user(username, password):
    dbx = get_db_x()
    user = dbx.execute("SELECT * FROM user WHERE username = ? ", username)
    dbx.close()        
    if not user: 
        return False
    elif check_password_hash(user["password"], password):
        return True