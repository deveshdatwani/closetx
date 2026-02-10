import os
from celery import Celery
BROKER = os.getenv("CELERY_BROKER") or f"redis://{os.getenv('REDIS_HOST','localhost')}:6379/0"
app = Celery("closetx_celery", broker=BROKER, backend=BROKER, include=['worker.task'])