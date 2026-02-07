import os
from celery import Celery
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

load_dotenv("model/.env")
logger.info("Initializing Celery app with Redis broker and backend")
HOST = os.getenv("REDIS_HOST", "localhost")
logger.info(HOST)

app = Celery("celery_app",
             broker=f"redis://{HOST}:6379/0",
             backend=f"redis://{HOST}:6379/0")