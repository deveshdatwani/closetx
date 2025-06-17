from celery import Celery


HOST = "localhost" 


app = Celery("celery_app",
             broker="redis://redis:6379/0",
             backend="redis://redis:6379/0")
