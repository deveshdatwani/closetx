from celery_app import app


@app.task()
def add(number1, number2):
    return int(number1 + number2)