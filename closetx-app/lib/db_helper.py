from flask import session, g, current_app, Response
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from PIL import Image
from base64 import encodebytes
import io, os
import uuid


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
DB connector should make repeated attempts to connect to the db and not give up on a single try
'''

def serve_response(data: str, status_code: int):
    response = Response(response=data, status=status_code)
    return response



def get_db_x():
    ATTEMPTS = 4
    try:
        while ATTEMPTS:
            current_app.logger.info("TRYING TO CONNECT TO MYSQL ENGINE")
            cnx = mysql.connector.connect(
                user='root',
                password='password',
                host='db',
                database='closetx',
                port=3306
            )
            g.db = cnx
            current_app.logger.info(f"MYSQL CONNECTOR SUCCESSFULLY CONNECTED TO DB AFTER {5-ATTEMPTS} ATTEMPT")
            ATTEMPTS -= 1
            if cnx: break
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
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, password))
            dbx.commit()
            crx.close()
            dbx.close()
        except mysql.connector.errors.IntegrityError:
            current_app.logger.error("DUPLICATE ENTRIES")            
            return False
        return True
    else:
        current_app.logger.error("COULD NOT CONNECT TO MYSQL")   
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
                return True       
            else:
                return None 
        except Exception as e:
            current_app.logger.error(e, "REDIRECTING")
            return False
    

def delete_user(username):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            user = crx.execute("SELECT * FROM user WHERE username = %s", (username,))
            if crx.fetchall():
                crx.execute("DELETE FROM user WHERE username = %s", (username,))
                dbx.commit()
                crx.close()
                dbx.close()
                return True
            else:
                current_app.logger.error("COULD NOT FIND USER")
                crx.close()
                dbx.close()
                return False
        except Exception as e:
            current_app.logger.error(e)
            return False
        

def post_apparel(userid, image_file):
    dbx = get_db_x()
    upload_folder = "./"
    apparel_uuid = str(uuid.uuid4())
    image_file_path = (os.path.join(upload_folder, apparel_uuid))
    image_file.save(image_file_path)
    import boto3
    bucket_name = 'closetx'
    # boto3.setup_default_session(
    # aws_access_key_id='your_access_key_id',
    # aws_secret_access_key='your_secret_access_key',
    # region_name='your_region')
    s3 = boto3.client('s3')
    with open(image_file_path, 'rb') as data:
        s3.upload_fileobj(data, bucket_name, f'{apparel_uuid}')
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            userid = crx.execute("INSERT INTO apparel (user, uri) VALUES (%s, %s)",(userid, apparel_uuid))
            dbx.commit()
            crx.close()
            dbx.close()
            return True
        except Exception as e:
            current_app.logger.error(e, "COULD NOT INSERT APPAREL INTO DB -- REDIRECTERING")
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
            else:
                return apparels
        except Exception as e:
            current_app.logger.error(e, "REDIRECTERING")
            return False        
        

def get_images(file_name):
    BASE_DIR = "./"
    file_path = os.path.join(BASE_DIR, file_name)
    pil_img = Image.open(file_path, mode='r') 
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') #
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 
    return encoded_img