import io, os
from PIL import Image
from time import time
import mysql.connector
from mysql.connector import errorcode
from flask import session, g, current_app
import torchvision.transforms as transforms


'''
SQL queries should not be string formatted. It is susceptible to SQL injections. Use ? instead. To work on when I can -- 09/24/2024
DB connector should make repeated attempts to connect to the db and not give up on a single try
'''


def get_db_x():
    password = os.getenv('DB_PASSWORD', 'password')
    db_host = os.getenv('DB_HOST', '127.0.0.1')
    db_port = os.getenv('DB_PORT', '3306')
    current_app.logger.info("Connecting to mysql sever")
    cnx = mysql.connector.connect(
                                user='root',
                                password=password,
                                host=db_host,
                                database='closetx',
                                port=db_port)
    current_app.logger.info(f"Successfully connected to mysql sever")
    return cnx


def inference(model, top, bottom):
    transform = transforms.ToTensor()
    top = transform(top)
    bottom = transform(bottom)
    top = top.unsqueeze(0)
    bottom = bottom.unsqueeze(0)
    start = time()
    score = model(bottom, top)
    end = time()
    total_time_taken = (end - start)
    print(f"Inference time {total_time_taken}") 
    return score