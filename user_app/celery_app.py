from celery import Celery


app = Celery("flask",
             broker="redis://redis:6379/0",
             backend="redis://redis:6379/0")


@app.task()
def add(number1, number2):
    return int(number1 + number2)
