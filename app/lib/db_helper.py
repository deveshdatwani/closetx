from flask import session, g, current_app
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from PIL import Image
from base64 import encodebytes
import io, os


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
DB connector should make repeated attempts to connect to the db and not give up on a single try
'''


def get_db_x():
    try:
        current_app.logger.info("TRYING TO CONNECT")
        cnx = mysql.connector.connect(
            user='root',
            password='password',
            host='127.0.0.1',
            database='closetx',
            port=3000
        )
        print("------- CONNNECTED -------")
        g.db = cnx
        current_app.logger.info("MYSQL CONNECTOR SUCCESSFULLY CONNECTED TO DB")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            current_app.logger.error("FAILED TO AUTHENTICATE MYSQL CONNECTOR")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            current_app.logger.error("DATABASE DOES NOT EXIST")
        else:
            current_app.logger.error(err)        
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
    print("TRYING TO CONNECT TO DB")
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
        

def post_apparel(userid, image_file, upload_folder):
    dbx = get_db_x()
    UPLOAD_FOLDER = upload_folder
    image_file_path = (os.path.join(UPLOAD_FOLDER, image_file.filename))
    image_file.save(image_file_path)
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            userid = crx.execute("INSERT INTO apparel (id, uri) VALUES (%s, %s)",(userid, image_file_path))
            print("ADDED APPAREL")
            dbx.commit()
            crx.close()
            dbx.close()
            return True
        except Exception as e:
            print(e, "COULD NOT INSERT APPAREL INTO DB -- REDIRECTERING")
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
                return apparels
        except Exception as e:
            print(e, "REDIRECTERING")
            return False        
        

def get_images(file_name):
    BASE_DIR = "./"
    file_path = os.path.join(BASE_DIR, file_name)
    pil_img = Image.open(file_path, mode='r') 
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') #
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 
    return encoded_img