import os
import cv2
import torch
import numpy as np
from PIL import Image
from time import time
import mysql.connector
from flask import current_app


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
    top = Image.open(top)
    top = np.array(top, np.float32)
    top = cv2.resize(top,(576, 576))
    top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
    bottom = Image.open(bottom)
    bottom = np.array(bottom, np.float32)
    bottom = cv2.resize(bottom,(576, 576))
    bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
    top = torch.permute(torch.from_numpy(top), (2, 0, 1))
    bottom = torch.permute(torch.from_numpy(bottom), (2, 0, 1))
    top = top.unsqueeze(0)
    bottom = bottom.unsqueeze(0)
    start_time = time()
    score = model(bottom, top)
    end_time = time()
    current_app.logger.info(f"Inferenced {top.shape[0]} batches in {end_time - start_time} seconds")
    return score