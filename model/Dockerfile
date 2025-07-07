FROM python:3.10

COPY ./user_app/celery_app.py /user_app/celery_app.py
COPY ./model /model
RUN mkdir /closet
WORKDIR /

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN pip install --upgrade pip
RUN apt-get install -y default-libmysqlclient-dev
RUN pip install -r /model/requirements.txt
RUN apt-get install -y default-mysql-client

CMD ["python3", "-m", "celery", "-A", "user_app.celery_app", "worker", "--pool", "solo", "--loglevel=debug"] 
