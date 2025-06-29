FROM python:3.10

COPY ./user_app/celery_app.py /celery_app/celery_app.py
COPY ./model /celery_app/model
WORKDIR /celery_app

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN pip install --upgrade pip
RUN apt-get install -y default-libmysqlclient-dev
RUN pip install -r model/requirements.txt
RUN apt-get install -y default-mysql-client

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=debug"] 
