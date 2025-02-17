import uuid
import boto3
import io, os
from PIL import Image
import mysql.connector
from base64 import encodebytes
from mysql.connector import errorcode
from flask import g, current_app, Response
from werkzeug.security import check_password_hash, generate_password_hash


def serve_response(data: str, status_code: int):
    response = Response(response=data, status=status_code)
    return response


def get_s3_boto_client():
    boto3.setup_default_session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                                aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
                                region_name='us-east-2')
    current_app.logger.info("S3 client connected")
    s3 = boto3.client('s3')
    return s3


def get_db_x():
    password = os.getenv('DB_PASSWORD', 'hello')
    db_host = os.getenv('DB_HOST', '172.17.0.1')
    db_port = os.getenv('DB_PORT', '3306')
    database = 'closetx'
    user = 'closetx'
    current_app.logger.info("Connecting to mysql sever")
    cnx = mysql.connector.connect(
        user=user,
        password=password,
        host=db_host,
        database=database,
        port=db_port)
    current_app.logger.info(f"Successfully connected to mysql")
    return cnx


def register_user(username: str, password: str, email: str) -> bool:
    dbx = get_db_x()
    crx = dbx.cursor()
    auth_string = generate_password_hash(password)
    crx.execute("INSERT INTO user (username, password, email) VALUES (%s, %s, %s)", (username, auth_string, email))
    dbx.commit()
    crx.close()
    dbx.close()
    return True


def login_user(username, password):
    dbx = get_db_x()
    current_app.logger.info("Matching password for user")
    crx = dbx.cursor()
    crx.execute("SELECT * FROM user WHERE username = %s", (username,))
    user = crx.fetchone()
    crx.close()
    dbx.close()  
    if check_password_hash(user[3], password): return user
    else: return None


def get_user(username):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("SELECT * FROM user WHERE username = %s", (username,))
    user = crx.fetchall()
    crx.close()
    dbx.close()     
    return user


def delete_user(username):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("DELETE FROM user WHERE username = %s", (username,))
    dbx.commit()
    crx.close()
    dbx.close()
    return True

# --------------------------------------- #        

def post_apparel(userid, image):
    dbx = get_db_x()
    apparel_uuid = str(uuid.uuid4())+".png"
    s3_client = get_s3_boto_client()
    image.save('./file.png')
    s3_client.upload_file('./file.png', 'closetx', apparel_uuid)
    os.remove('./file.png')
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
    s3 = get_s3_boto_client()
    if s3:
        try:
            with open('file', 'wb') as data:
                s3.download_fileobj('closetx', uri, data)
        except Exception as e:
            current_app.logger.error(e) 
            current_app.logger.warning("No resource found for given uri")
            data = "No apparel found"
            return serve_response(data=data, status_code=403)
    apparel_image = Image.open('./file')
    img_io = io.BytesIO()
    apparel_image.save(img_io, 'PNG')
    img_io.seek(0)
    os.remove('./file')
    return img_io 
    

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


def delete_apparel(userid, uri):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("DELETE FROM apparel WHERE uri = %s", (uri,))
            dbx.commit()
            crx.close() 
            dbx.close()
        except Exception as e:
            current_app.logger.error("Could not delete apparel")
            current_app.logger.error(e)
            return False
        return True
    
    
def delete_closet(userid):
    dbx = get_db_x()
    if dbx and dbx.is_connected():
        try:
            crx = dbx.cursor()
            crx.execute("DELETE FROM apparel WHERE user = %s", (userid,))
            dbx.commit()
            crx.close() 
            dbx.close()
        except Exception as e:
            current_app.logger.error("Could not delete closet")
            current_app.logger.error(e)
            return False
        return True