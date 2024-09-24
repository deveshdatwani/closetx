from flask import session, g
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import check_password_hash, generate_password_hash
import logging


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
'''


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
            print("FAILED TO AUTHENTICATE MYSQL CONNECTOR")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("DATABASE DOES NOT EXIST")
        else:
            print(err)
        
        return None
    
    return cnx


def register_user(username, password):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, password))
            dbx.commit()
            crx.close()
            dbx.close()
        except mysql.connector.errors.IntegrityError:
            print("DUPLICATE ENTRIES")
            
            return False
        
        return True
    else:
        print("COULD NOT CONNECT") 
        
        return False


def login_user(username, password):
    try:
        dbx = get_db_x()
        crx = dbx.cursor()
        crx.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = crx.fetchone()
        crx.close()
        dbx.close()    

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
    

def delete_user(username):
    try:
        dbx = get_db_x()
        crx = dbx.cursor()
        crx.execute("DELETE FROM user WHERE username = %s", (username,))
        dbx.commit()
        crx.close()
        dbx.close()
        print("DELETED USER")
    
        return True
    except Exception as e:
        print(e)
        return False