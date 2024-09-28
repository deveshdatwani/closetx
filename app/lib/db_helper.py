from flask import session, g
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import check_password_hash, generate_password_hash
import logging


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
DB connector should make repeated attempts to connect to the db and not give up on a single try
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
        print("MYSQL CONNECTOR SUCCESSFULLY CONNECTED TO DB")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("FAILED TO AUTHENTICATE MYSQL CONNECTOR")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("DATABASE DOES NOT EXIST")
        else:
            print(err)        
        return None
    return cnx


def get_user(username):
    dbx = get_db_x()
    crx = crx = dbx.cursor()
    crx.execute("SELECT * FROM user WHERE username = %s", (username,))
    user = crx.fetchall()
    crx.close()
    dbx.close()     
    return user


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
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("SELECT * FROM user WHERE username = %s", (username,))
            user = crx.fetchone()
            crx.close()
            dbx.close()    
            if not user:
                return "USER NOT FOUND"        
            # elif check_password_hash(password, user[3]): need to check for password hash instead of string, 
            elif user[3] == password:            
                return "CORRECT PASSWORD"        
            else:
                return None 
        except Exception as e:
            print(e, "redirecting")
            return False
    

def delete_user(username):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            user = crx.execute("SELECT * FROM user WHERE username = %s", (username,))
            if crx.fetchall():
                crx.execute("DELETE FROM user WHERE username = %s", (username,))
                print("DELETED USER")
                dbx.commit()
                crx.close()
                dbx.close()
                return True
            else:
                print("COULD NOT FIND USER")
                crx.close()
                dbx.close()
                return False
        except Exception as e:
            print(e)
            return False
        

def post_apparel(userid, uri):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            userid = crx.execute("INSERT INTO apparel (id, uri) VALUES (%s, %s)",(userid, uri))
            print("ADDED APPAREL")
            dbx.commit()
            crx.close()
            dbx.close()
            return True
        except Exception as e:
            print(e, "REDIRECTERING")
    else:
        return False


def get_apparel(userid):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            userds = crx.execute("SELECT * FROM apparel WHERE id = %s", (userid,))
            apparels = crx.fetchall()
            crx.close()
            dbx.close()    
            if not apparels:
                return "NO APPARELS FOUND"        
            # currently only sends file paths of images of apparel
            else:
                return "TRUE"
        except Exception as e:
            print(e, "REDIRECTERING")
            return False        