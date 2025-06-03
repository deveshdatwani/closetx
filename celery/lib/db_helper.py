import os
import cv2
import torch
import numpy as np
from PIL import Image
from time import time
import mysql.connector
from flask import current_app


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