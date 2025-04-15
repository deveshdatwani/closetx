import uuid
import boto3
import io, os
import requests
from PIL import Image
import mysql.connector
from base64 import encodebytes
from mysql.connector import errorcode
from flask import g, current_app, Response
from werkzeug.security import check_password_hash, generate_password_hash
from mysql import connector
from io import BytesIO
from matplotlib import pyplot as plt

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
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '3306')
    database = 'closetx'
    user = 'closetx'
    current_app.logger.info("Connecting to mysql sever")
    conn = mysql.connector.connect(
        database=database,   
        user=user,  
        password=password,  
        host=db_host,
        port=db_port
    )
    current_app.logger.info(f"Successfully connected to mysql")
    return conn


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


def post_apparel(userid, image):
    response = requests.post("http://localhost:6000/model/segment", files={"image":image})
    image = Image.open(BytesIO(response.content))
    dbx = get_db_x()
    apparel_uuid = str(uuid.uuid4()) + ".png"
    s3_client = get_s3_boto_client()
    image.save('./file.png')
    s3_client.upload_file('./file.png', 'closetx-images', apparel_uuid)
    os.remove('./file.png')
    crx = dbx.cursor()
    userid = crx.execute("INSERT INTO apparel (user, uri) VALUES (%s, %s)",(userid, apparel_uuid))
    dbx.commit()
    crx.close() 
    dbx.close()
    current_app.logger.info("Inserted image into s3")
    current_app.logger.info("Apparel URI inserted into DB")
    return True
        

def get_apparel(uri):
    s3 = get_s3_boto_client()
    with open(uri, 'wb') as data:
        s3.download_fileobj('closetx-images', uri, data)
    apparel_image = Image.open(uri)
    img_io = io.BytesIO()
    apparel_image.save(img_io, 'PNG')
    img_io.seek(0)
    os.remove(uri)
    current_app.logger.info("Fetched image")
    return img_io 


def get_user_apparels(userid):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("SELECT uri FROM apparel WHERE user = %s", (userid,))
    apparel_ids = crx.fetchall()
    crx.close()
    dbx.close()
    return apparel_ids


def delete_apparel(userid, uri):
    dbx = get_db_x()
    crx = dbx.cursor()  
    crx.execute("DELETE FROM apparel WHERE uri = %s", (uri,))
    dbx.commit()
    crx.close() 
    dbx.close()
    return True
    

def delete_closet(userid):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("DELETE FROM apparel WHERE user = %s", (userid,))
    dbx.commit()
    crx.close() 
    dbx.close()
    return True


def rgb_from_img(image):
    json_response = requests.post("http://127.0.0.1:5001/model/get-color-from-apparel", files={"image":image})
    return json_response
