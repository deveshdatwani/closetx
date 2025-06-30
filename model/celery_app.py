from celery import Celery


HOST = "localhost" 


app = Celery("model_celery",
             broker="redis://127.0.0.1:6379/0",
             backend="redis://127.0.0.1:6379/0")


@app.task()
def add(number1, number2):
    return int(number1 + number2)