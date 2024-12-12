import uuid
import boto3
import io, os
from PIL import Image
import mysql.connector
from base64 import encodebytes
from mysql.connector import errorcode
from flask import g, current_app, Response, send_file
from werkzeug.security import check_password_hash, generate_password_hash


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
DB connector should make repeated attempts to connect to the db and not give up on a single try
'''


def get_s3_boto_client():
    try:
        boto3.setup_default_session(aws_access_key_id=os.getenv('aws_access_key_id'),
                                    aws_secret_access_key=os.getenv('aws_secret_access_key_id'),
                                    region_name='us-east-2')
        current_app.logger.debug("S3 client connected")
    except Exception as e:
        current_app.logger.error("Cannot not connect to S3")
        current_app.logger.error(e)
        return None
    s3 = boto3.client('s3')
    return s3


def serve_response(data: str, status_code: int):
    response = Response(response=data, status=status_code)
    return response


def get_db_x():
    attempts = 4
    try:
        while attempts:
            current_app.logger.debug("Connecting to mysql sever")
            cnx = mysql.connector.connect(
                user='root',
                password='password',
                host='127.0.0.1',
                database='closetx',
                port=3307)
            g.db = cnx
            current_app.logger.info(f"Successfully connected to mysql sever after {5-attempts} attempts")
            attempts -= 1
            if cnx: break
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            current_app.logger.error("Failed to authenticate client on mysql server")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            current_app.logger.error("Database closetx does not exist")
        else:
            current_app.logger.error(err)        
        return None
    return cnx


def get_user(username):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("SELECT * FROM user WHERE username = %s", (username,))
    user = crx.fetchall()
    crx.close()
    dbx.close()     
    return user


def register_user(username, password, email):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            auth_string = generate_password_hash(password)
            crx.execute("INSERT INTO user (username, password, email) VALUES (%s, %s, %s)", (username, auth_string, email))
            dbx.commit()
            crx.close()
            dbx.close()
        except mysql.connector.errors.IntegrityError:
            current_app.logger.error("This username already exists or email")            
            current_app.logger.error("User already exists")            
            return False
        return True
    else:
        current_app.logger.error("Could not connect to mysql engine")   
        current_app.logger.error("Could not connect to mysql server")   
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
                return False
            elif check_password_hash(user[3], password):
                current_app.logger.info("User password matched")
                return user      
            else:
                return None 
        except Exception as e:
            current_app.logger.error(e)
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
                current_app.logger.error("Could not find user")
                crx.close()
                dbx.close()
                return False
        except Exception as e:
            current_app.logger.error(e)
            return False
        

def post_apparel(userid, image):
    dbx = get_db_x()
    apparel_uuid = str(uuid.uuid4())
    uri_path = os.path.join("./", f'{apparel_uuid}.png')
    image.save(uri_path)
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            userid = crx.execute("INSERT INTO apparel (user, uri) VALUES (%s, %s)",(userid, apparel_uuid))
            dbx.commit()
            crx.close() 
            dbx.close()
            return True
        except Exception as e:
            current_app.logger.error(e, "Could not insert apparel into closet")
    else:
        return False


def get_apparel(uri):
    try:
        uri_path = os.path.join("./", uri)
        image_file = io.BytesIO(Image.open(uri_path))
    except Exception as e:
        current_app.logger.error(e) 
        current_app.logger.warning("No resource found for given uri")
        data = "No apparel found"
        return serve_response(data=data, status_code=403)
    return send_file(io.BytesIO(image_file), mimetype='image/png')
    

def get_images(file_name):
    BASE_DIR = "./"
    file_path = os.path.join(BASE_DIR, file_name)
    pil_img = Image.open(file_path, mode='r') 
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') #
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 
    return encoded_img


def get_user_apparels(userid):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("SELECT uri FROM apparel WHERE user = %s", (userid,))
            apparel_ids = crx.fetchall()
            crx.close()
            dbx.close()
            return apparel_ids
        except Exception as e:
            current_app.logger.error(e)