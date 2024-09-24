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
            dbx.cursor().execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, password)).fetch()
            dbx.commit()
            dbx.cursor.close()
            dbx.close()
        except mysql.connector.errors.IntegrityError:
            print("Duplicate entries")
            return False
        
        return True
    else:
        print("Could not connect") 
        
        return False


def login_user(username, password):
    try:
        dbx = get_db_x()
        crx = dbx.cursor()
        crx.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = crx.fetchone()
        crx.close()
        dbx.close()    

        print(user[3] == password)

        if not user:
            return "USER NOT FOUND"
        
        # elif check_password_hash(password, user[3]):
        elif user[3] == password:
            return "CORRECT PASSWORD"
        
        else:
            return None 

    except Exception as e:
        print(e, "redirecting")
        return False