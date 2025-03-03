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
    db_host = os.getenv('DB_HOST', '127.0.0.1')
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


def post_apparel(userid, image):
    dbx = get_db_x()
    apparel_uuid = str(uuid.uuid4()) + ".png"
    s3_client = get_s3_boto_client()
    image.save('./file.png')
    s3_client.upload_file('./file.png', 'closetx', apparel_uuid)
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
    with open('file', 'wb') as data:
        s3.download_fileobj('closetx', uri, data)
    apparel_image = Image.open('./file')
    img_io = io.BytesIO()
    apparel_image.save(img_io, 'PNG')
    img_io.seek(0)
    os.remove('./file')
    return img_io 


def get_user_apparels(userid):
    dbx = get_db_x()
    crx = dbx.cursor()
    crx.execute("SELECT uri FROM apparel WHERE user = %s", (userid,))
    apparel_ids = crx.fetchall()
    crx.close()
    dbx.close()
    return apparel_ids


def cached_apparel(img_path:str):
    apparel_image = Image.open(os.path.join(current_app.config['CACHE_DIR'], img_path + '.png'))
    img_io = io.BytesIO()
    apparel_image.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def rgb_from_img(image):
    json_response = requests.post("http://127.0.0.1:5001/model/get-color-from-apparel", files={"image":image})
    return json_response


def match_all(img, global_dict):
    response = {}    
    img_colors = requests.post("http://127.0.0.1:5001/model/get-color-from-apparel", files={"image":img}).json()
    img_colors = img_colors["r"], img_colors["g"], img_colors["b"] 
    for closet_apparel in global_dict:
        closet_color = global_dict[closet_apparel]["r"], global_dict[closet_apparel]["g"], global_dict[closet_apparel]["b"]
        payload = {"r1":img_colors[0], "g1":img_colors[1], "b1":img_colors[2], "r2":closet_color[0], "g2":closet_color[1], "b2":closet_color[2]}
        match_score = requests.post("http://127.0.0.1:5001/model/match", data=payload)
        response[closet_apparel] = int(match_score.content)
    return response

# --------- #


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