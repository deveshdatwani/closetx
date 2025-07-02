import os
from celery import Celery

config = os.getenv("USER_APP_ENV", "prod")
if config == "prod": HOST = "redis"
else: HOST = "127.0.0.1"



app = Celery("flask",
             broker=f"redis://{HOST}:6379/0",
             backend=f"redis://{HOST}:6379/0")


@app.task()
def add(number1, number2):
    return int(number1 + number2)
